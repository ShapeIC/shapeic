use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek};

use ndarray::{ArrayD, IxDyn};
use ndarray_npy::npy::header::{Header, Layout};
use serde_pickle::{DeOptions, HashableValue, Value};
use zip::ZipArchive;

use crate::{Axis, DeviceLut, LookupTable, LutArray, LutError, LutMetadata};

const LOOKUP_TABLE_MEMBER: &str = "lookup_table.npy";
const ROOT_METADATA_KEYS: [&str; 4] = [
    "description",
    "simulator",
    "parameter_names",
    "device_parameters",
];
const MODEL_METADATA_KEYS: [&str; 4] = [
    "model_name",
    "parameter_names",
    "device_parameters",
    "description",
];

pub(crate) fn read_lookup_table<R>(reader: R) -> Result<LookupTable, LutError>
where
    R: Read + Seek,
{
    let mut archive = ZipArchive::new(reader)?;
    let mut member = archive.by_name(LOOKUP_TABLE_MEMBER)?;
    let header = Header::from_reader(&mut member)?;
    validate_outer_header(&header)?;

    let value = serde_pickle::value_from_reader(
        &mut member,
        DeOptions::new()
            .keep_restore_state()
            .replace_recursive_structures(),
    )?;
    let root = unwrap_lookup_table(value)?;
    parse_root(root)
}

fn validate_outer_header(header: &Header) -> Result<(), LutError> {
    if header.layout != Layout::Standard {
        return Err(LutError::unsupported(
            LOOKUP_TABLE_MEMBER,
            "the outer object array must use C order",
        ));
    }
    if !header.shape.is_empty() {
        return Err(LutError::schema(
            LOOKUP_TABLE_MEMBER,
            format!(
                "expected a scalar object array, found shape {:?}",
                header.shape
            ),
        ));
    }

    let descriptor = header
        .type_descriptor
        .as_string()
        .ok_or_else(|| LutError::unsupported(LOOKUP_TABLE_MEMBER, "non-string dtype descriptor"))?;
    if !matches!(descriptor.as_str(), "|O" | "|O8" | "<O8" | "=O8") {
        return Err(LutError::unsupported(
            LOOKUP_TABLE_MEMBER,
            format!("expected an object dtype, found '{descriptor}'"),
        ));
    }
    Ok(())
}

fn unwrap_lookup_table(value: Value) -> Result<BTreeMap<HashableValue, Value>, LutError> {
    if let Value::Dict(root) = value {
        return Ok(root);
    }

    let state = ndarray_state(value, LOOKUP_TABLE_MEMBER)?;
    if !state.shape.is_empty() || state.fortran {
        return Err(LutError::schema(
            LOOKUP_TABLE_MEMBER,
            "outer ndarray state is not a C-order scalar",
        ));
    }
    match state.payload {
        Value::List(mut values) if values.len() == 1 => match values.pop() {
            Some(Value::Dict(root)) => Ok(root),
            _ => Err(LutError::schema(
                LOOKUP_TABLE_MEMBER,
                "object scalar does not contain a dictionary",
            )),
        },
        _ => Err(LutError::schema(
            LOOKUP_TABLE_MEMBER,
            "object scalar payload must be a one-element list",
        )),
    }
}

fn parse_root(root: BTreeMap<HashableValue, Value>) -> Result<LookupTable, LutError> {
    let mut root = string_keyed_dict(root, "lookup_table")?;
    let metadata = LutMetadata {
        description: take_optional_string(&mut root, "description", "lookup_table")?,
        simulator: take_optional_string(&mut root, "simulator", "lookup_table")?,
    };
    let root_parameter_names =
        take_optional_string_list(&mut root, "parameter_names", "lookup_table")?
            .unwrap_or_default();
    let root_device_parameters =
        take_optional_number_dict(&mut root, "device_parameters", "lookup_table")?
            .unwrap_or_default();

    let mut models = BTreeMap::new();
    for (name, value) in root {
        if ROOT_METADATA_KEYS.contains(&name.as_str()) {
            continue;
        }
        let Value::Dict(dict) = value else {
            continue;
        };
        let model = string_keyed_dict(dict, format!("model '{name}'"))?;
        if !Axis::ALL
            .iter()
            .all(|axis| model.contains_key(axis.as_str()))
        {
            continue;
        }
        let device = parse_model(
            name.clone(),
            model,
            &root_parameter_names,
            &root_device_parameters,
        )?;
        models.insert(name, device);
    }

    if models.is_empty() {
        return Err(LutError::schema(
            "lookup_table",
            "no model with length, vbs, vgs and vds axes was found",
        ));
    }

    // NumPy reuses these metadata objects. serde-pickle deliberately replaces
    // a repeated memo reference after its original value has been consumed,
    // so recover missing entries from any intact model copy.
    let shared_device_parameters = models
        .values()
        .flat_map(|model| model.device_parameters.iter())
        .fold(root_device_parameters, |mut parameters, (name, value)| {
            parameters.entry(name.clone()).or_insert(*value);
            parameters
        });
    for model in models.values_mut() {
        for (name, value) in &shared_device_parameters {
            model
                .device_parameters
                .entry(name.clone())
                .or_insert(*value);
        }
    }
    Ok(LookupTable { metadata, models })
}

fn parse_model(
    name: String,
    mut model: BTreeMap<String, Value>,
    root_parameter_names: &[String],
    root_device_parameters: &BTreeMap<String, f64>,
) -> Result<DeviceLut, LutError> {
    let context = format!("model '{name}'");
    let declared_names = take_optional_string_list(&mut model, "parameter_names", &context)?
        .unwrap_or_else(|| root_parameter_names.to_vec());
    let device_parameters = take_optional_number_dict(&mut model, "device_parameters", &context)?
        .unwrap_or_else(|| root_device_parameters.clone());
    model.remove("model_name");
    model.remove("description");

    let mut axes = std::array::from_fn(|_| Vec::new());
    for axis in Axis::ALL {
        let value = model.remove(axis.as_str()).ok_or_else(|| {
            LutError::schema(&context, format!("missing '{}' axis", axis.as_str()))
        })?;
        let array = decode_numeric_array(value, &format!("{context}.{}", axis.as_str()))?;
        axes[axis.index()] = decode_axis(array, &name, axis)?;
    }

    let expected_shape = [axes[0].len(), axes[1].len(), axes[2].len(), axes[3].len()];
    let mut parameters = BTreeMap::new();
    let mut parameter_names = Vec::new();
    let mut seen = BTreeSet::new();

    for parameter in declared_names {
        if !seen.insert(parameter.clone()) {
            continue;
        }
        let Some(value) = model.remove(&parameter) else {
            continue;
        };
        let array = decode_numeric_array(value, &format!("{context}.{parameter}"))?;
        validate_parameter_shape(&array, expected_shape, &context, &parameter)?;
        parameter_names.push(parameter.clone());
        parameters.insert(parameter, array);
    }

    for (parameter, value) in model {
        if MODEL_METADATA_KEYS.contains(&parameter.as_str()) || seen.contains(&parameter) {
            continue;
        }
        if !matches!(value, Value::Tuple(_)) {
            continue;
        }
        let array = decode_numeric_array(value, &format!("{context}.{parameter}"))?;
        validate_parameter_shape(&array, expected_shape, &context, &parameter)?;
        parameter_names.push(parameter.clone());
        parameters.insert(parameter, array);
    }

    if parameters.is_empty() {
        return Err(LutError::schema(
            &context,
            "model contains no numeric parameters",
        ));
    }

    Ok(DeviceLut {
        name,
        axes,
        parameters,
        parameter_names,
        device_parameters,
    })
}

fn decode_axis(array: LutArray, model: &str, axis: Axis) -> Result<Vec<f64>, LutError> {
    if array.shape().len() != 1 || array.is_empty() {
        return Err(LutError::schema(
            format!("model '{model}'.{axis}"),
            format!(
                "axis must be a non-empty vector, found shape {:?}",
                array.shape()
            ),
        ));
    }
    let values = array.into_f64_vec();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(LutError::NonFinite {
            model: model.to_owned(),
            context: format!("'{axis}' axis"),
        });
    }
    let ascending = values.windows(2).all(|window| window[0] < window[1]);
    let descending = values.windows(2).all(|window| window[0] > window[1]);
    if values.len() > 1 && !ascending && !descending {
        return Err(LutError::schema(
            format!("model '{model}'.{axis}"),
            "axis values must be strictly monotonic",
        ));
    }
    Ok(values)
}

fn validate_parameter_shape(
    array: &LutArray,
    expected: [usize; 4],
    model_context: &str,
    parameter: &str,
) -> Result<(), LutError> {
    if array.shape() == [1] || array.shape() == expected {
        return Ok(());
    }
    Err(LutError::schema(
        format!("{model_context}.{parameter}"),
        format!(
            "parameter shape {:?} must be [1] or {expected:?}",
            array.shape()
        ),
    ))
}

struct NdarrayState {
    shape: Vec<usize>,
    dtype: Value,
    fortran: bool,
    payload: Value,
}

fn ndarray_state(value: Value, context: &str) -> Result<NdarrayState, LutError> {
    let Value::Tuple(state) = value else {
        return Err(LutError::schema(context, "expected an ndarray state tuple"));
    };
    if state.len() != 5 {
        return Err(LutError::schema(
            context,
            format!("ndarray state must have five fields, found {}", state.len()),
        ));
    }

    let mut fields = state.into_iter();
    let version = fields
        .next()
        .ok_or_else(|| LutError::schema(context, "missing ndarray state version"))?;
    let shape = value_to_shape(
        fields
            .next()
            .ok_or_else(|| LutError::schema(context, "missing ndarray shape"))?,
        context,
    )?;
    let dtype = fields
        .next()
        .ok_or_else(|| LutError::schema(context, "missing ndarray dtype"))?;
    let fortran = match fields
        .next()
        .ok_or_else(|| LutError::schema(context, "missing ndarray order flag"))?
    {
        Value::Bool(value) => value,
        _ => return Err(LutError::schema(context, "invalid ndarray order flag")),
    };
    let payload = fields
        .next()
        .ok_or_else(|| LutError::schema(context, "missing ndarray payload"))?;
    match version {
        Value::I64(1) => {}
        _ => {
            return Err(LutError::unsupported(
                context,
                "unsupported ndarray state version",
            ));
        }
    }
    Ok(NdarrayState {
        shape,
        dtype,
        fortran,
        payload,
    })
}

fn value_to_shape(value: Value, context: &str) -> Result<Vec<usize>, LutError> {
    let Value::Tuple(dimensions) = value else {
        return Err(LutError::schema(context, "ndarray shape is not a tuple"));
    };
    dimensions
        .into_iter()
        .map(|dimension| match dimension {
            Value::I64(value) => usize::try_from(value)
                .map_err(|_| LutError::schema(context, "negative or oversized ndarray dimension")),
            _ => Err(LutError::schema(context, "non-integer ndarray dimension")),
        })
        .collect()
}

fn decode_numeric_array(value: Value, context: &str) -> Result<LutArray, LutError> {
    let state = ndarray_state(value, context)?;
    if state.fortran {
        return Err(LutError::unsupported(
            context,
            "Fortran-order arrays are not supported",
        ));
    }
    let Value::Bytes(bytes) = state.payload else {
        return Err(LutError::unsupported(
            context,
            "numeric ndarray payload is not bytes",
        ));
    };

    let element_count = state.shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| LutError::schema(context, "ndarray element count overflow"))
    })?;
    if element_count == 0 {
        return Err(LutError::unsupported(
            context,
            "empty numeric arrays are not supported",
        ));
    }

    let dtype = dtype_hint(&state.dtype);
    if dtype.big_endian {
        return Err(LutError::unsupported(
            context,
            "big-endian arrays are not supported",
        ));
    }
    if let Some(name) = &dtype.unsupported {
        return Err(LutError::unsupported(
            context,
            format!("unsupported NumPy dtype '{name}'"),
        ));
    }

    let item_size = bytes
        .len()
        .checked_div(element_count)
        .ok_or_else(|| LutError::schema(context, "invalid ndarray byte length"))?;
    if item_size * element_count != bytes.len() {
        return Err(LutError::schema(
            context,
            "ndarray byte length does not match its shape",
        ));
    }

    match (dtype.bits, item_size) {
        (Some(32) | None, 4) => {
            let values = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let array = ArrayD::from_shape_vec(IxDyn(&state.shape), values)
                .map_err(|error| LutError::schema(context, error.to_string()))?;
            Ok(LutArray::F32(array))
        }
        (Some(64) | None, 8) => {
            let values = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            let array = ArrayD::from_shape_vec(IxDyn(&state.shape), values)
                .map_err(|error| LutError::schema(context, error.to_string()))?;
            Ok(LutArray::F64(array))
        }
        (Some(bits), size) => Err(LutError::schema(
            context,
            format!("dtype f{bits} conflicts with {size}-byte elements"),
        )),
        (None, size) => Err(LutError::unsupported(
            context,
            format!("cannot infer floating-point dtype from {size}-byte elements"),
        )),
    }
}

#[derive(Default)]
struct DtypeHint {
    bits: Option<u8>,
    big_endian: bool,
    unsupported: Option<String>,
}

fn dtype_hint(value: &Value) -> DtypeHint {
    let mut strings = Vec::new();
    collect_strings(value, &mut strings);
    let mut hint = DtypeHint::default();
    for value in strings {
        let lower = value.to_ascii_lowercase();
        if lower.starts_with('>') || lower == ">" {
            hint.big_endian = true;
        }
        if lower.contains("f4") || lower.contains("float32") {
            hint.bits = Some(32);
        } else if lower.contains("f8") || lower.contains("float64") {
            hint.bits = Some(64);
        } else if lower.starts_with(['i', 'u', 'c', 'b'])
            && lower.chars().any(|character| character.is_ascii_digit())
        {
            hint.unsupported = Some(value.clone());
        }
    }
    hint
}

fn collect_strings<'a>(value: &'a Value, strings: &mut Vec<&'a String>) {
    match value {
        Value::String(value) => strings.push(value),
        Value::Tuple(values) | Value::List(values) => {
            for value in values {
                collect_strings(value, strings);
            }
        }
        _ => {}
    }
}

fn string_keyed_dict(
    dict: BTreeMap<HashableValue, Value>,
    context: impl AsRef<str>,
) -> Result<BTreeMap<String, Value>, LutError> {
    let context = context.as_ref();
    dict.into_iter()
        .map(|(key, value)| match key {
            HashableValue::String(key) => Ok((key, value)),
            _ => Err(LutError::schema(
                context,
                "dictionary contains a non-string key",
            )),
        })
        .collect()
}

fn take_optional_string(
    values: &mut BTreeMap<String, Value>,
    key: &str,
    context: &str,
) -> Result<Option<String>, LutError> {
    match values.remove(key) {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(LutError::schema(
            context,
            format!("'{key}' must be a string"),
        )),
    }
}

fn take_optional_string_list(
    values: &mut BTreeMap<String, Value>,
    key: &str,
    context: &str,
) -> Result<Option<Vec<String>>, LutError> {
    let Some(value) = values.remove(key) else {
        return Ok(None);
    };
    let Value::List(values) = value else {
        return Err(LutError::schema(context, format!("'{key}' must be a list")));
    };
    values
        .into_iter()
        .filter_map(|value| match value {
            Value::String(value) => Some(Ok(value)),
            // serde-pickle uses None only for a repeated memo reference while
            // resolving NumPy's object graph. Numeric arrays are discovered
            // independently below, so omitting this duplicate is lossless.
            Value::None => None,
            _ => Some(Err(LutError::schema(
                context,
                format!("'{key}' contains a non-string value"),
            ))),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn take_optional_number_dict(
    values: &mut BTreeMap<String, Value>,
    key: &str,
    context: &str,
) -> Result<Option<BTreeMap<String, f64>>, LutError> {
    let Some(value) = values.remove(key) else {
        return Ok(None);
    };
    let Value::Dict(values) = value else {
        return Err(LutError::schema(
            context,
            format!("'{key}' must be a dictionary"),
        ));
    };
    values
        .into_iter()
        .filter_map(|(name, value)| match (name, value) {
            (_, Value::None) | (HashableValue::None, _) => None,
            (HashableValue::String(name), value) => Some(Ok((name, value))),
            _ => Some(Err(LutError::schema(
                format!("{context}.{key}"),
                "device parameter dictionary contains a non-string key",
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|(name, value)| {
            let number = match value {
                Value::F64(value) => value,
                Value::I64(value) => value as f64,
                _ => {
                    return Err(LutError::schema(
                        format!("{context}.{key}.{name}"),
                        "device parameter must be numeric",
                    ));
                }
            };
            if !number.is_finite() {
                return Err(LutError::NonFinite {
                    model: context.to_owned(),
                    context: format!("device parameter '{name}'"),
                });
            }
            Ok((name, number))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map(Some)
}
