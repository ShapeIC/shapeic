use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Cursor, Read, Seek};
use std::path::Path;

use ndarray::{ArrayD, Ix5};
use ndarray_npy::ReadNpyExt;
use serde::Deserialize;
use zip::ZipArchive;

use crate::LayoutError;
use crate::model::{
    DeviceCapacitanceCorrection, PhysicalLookupTable, PhysicalMetadata, PhysicalPrimitive,
};
use crate::validation::{validate_axis, validate_device_correction_grid, validate_grid};

const MANIFEST_MEMBER: &str = "manifest.json";
const FORMAT: &str = "shapeic-physical-lut";
const SUPPORTED_VERSIONS: [u32; 2] = [1, 2];
const AXIS_ORDER: [&str; 3] = ["length", "finger_width", "nf"];
const CORRECTION_AXIS_ORDER: [&str; 6] = ["length", "finger_width", "nf", "vbs", "vgs", "vds"];
const CORRECTION_DEFINITION: &str = "pex_mos_only_minus_aggregate_compact_model";

impl PhysicalLookupTable {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LayoutError> {
        let file = File::open(path)?;
        Self::from_reader(file)
    }

    pub fn from_reader<R: Read + Seek>(reader: R) -> Result<Self, LayoutError> {
        read_archive(reader)
    }
}

#[derive(Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    pdk: String,
    layout_policy: String,
    generator: Option<String>,
    primitives: Vec<PrimitiveManifest>,
}

#[derive(Deserialize)]
struct PrimitiveManifest {
    name: String,
    port_order: Vec<String>,
    axis_order: Vec<String>,
    axes: Vec<Member>,
    conductance: String,
    capacitance: String,
    device_capacitance_correction: Option<DeviceCorrectionManifest>,
}

#[derive(Deserialize)]
struct DeviceCorrectionManifest {
    definition: String,
    axis_order: Vec<String>,
    axes: Vec<Member>,
    capacitance: String,
}

#[derive(Deserialize)]
struct Member {
    name: String,
    path: String,
}

fn read_archive<R: Read + Seek>(reader: R) -> Result<PhysicalLookupTable, LayoutError> {
    let mut archive = ZipArchive::new(reader)?;
    let manifest = {
        let mut member = archive.by_name(MANIFEST_MEMBER)?;
        let mut bytes = Vec::new();
        member.read_to_end(&mut bytes)?;
        serde_json::from_slice::<Manifest>(&bytes)?
    };
    if manifest.format != FORMAT || !SUPPORTED_VERSIONS.contains(&manifest.version) {
        return Err(LayoutError::schema(
            MANIFEST_MEMBER,
            format!(
                "expected format '{FORMAT}' with version in {SUPPORTED_VERSIONS:?}, found '{}' version {}",
                manifest.format, manifest.version
            ),
        ));
    }
    if manifest.pdk.is_empty() || manifest.layout_policy.is_empty() {
        return Err(LayoutError::schema(
            MANIFEST_MEMBER,
            "pdk and layout_policy must not be empty",
        ));
    }
    if manifest.primitives.is_empty() {
        return Err(LayoutError::schema(
            MANIFEST_MEMBER,
            "primitives must not be empty",
        ));
    }

    let mut primitives = BTreeMap::new();
    let mut used_paths = BTreeSet::new();
    let format_version = manifest.version;
    for primitive in manifest.primitives {
        let context = format!("primitive '{}'", primitive.name);
        if primitive.name.is_empty() || primitive.port_order.len() < 2 {
            return Err(LayoutError::schema(
                &context,
                "name must be non-empty and at least two ports are required",
            ));
        }
        if primitive.port_order.iter().any(String::is_empty)
            || primitive.port_order.iter().collect::<BTreeSet<_>>().len()
                != primitive.port_order.len()
        {
            return Err(LayoutError::schema(
                &context,
                "port_order contains empty or duplicate names",
            ));
        }
        if primitive
            .axis_order
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != AXIS_ORDER
        {
            return Err(LayoutError::schema(
                &context,
                format!("axis_order must be {AXIS_ORDER:?}"),
            ));
        }
        if primitive.axes.len() != AXIS_ORDER.len()
            || primitive
                .axes
                .iter()
                .map(|axis| axis.name.as_str())
                .ne(AXIS_ORDER)
        {
            return Err(LayoutError::schema(&context, "axes must follow axis_order"));
        }
        let mut axes = Vec::with_capacity(3);
        for member in &primitive.axes {
            register_path(&member.path, &mut used_paths, &context)?;
            let array = read_array(&mut archive, &member.path)?;
            if array.ndim() != 1 {
                return Err(LayoutError::schema(
                    &context,
                    format!("axis '{}' must be one-dimensional", member.name),
                ));
            }
            let values = array.into_raw_vec_and_offset().0;
            validate_axis(&values, &context, &member.name)?;
            axes.push(values);
        }
        if axes[0].iter().any(|value| *value <= 0.0) || axes[1].iter().any(|value| *value <= 0.0) {
            return Err(LayoutError::schema(
                &context,
                "length and finger_width axes must be positive",
            ));
        }
        if axes[2]
            .iter()
            .any(|value| *value < 1.0 || *value > f64::from(u32::MAX) || value.fract() != 0.0)
        {
            return Err(LayoutError::schema(
                &context,
                "nf axis must contain positive integers",
            ));
        }
        register_path(&primitive.conductance, &mut used_paths, &context)?;
        register_path(&primitive.capacitance, &mut used_paths, &context)?;
        let conductance = read_array(&mut archive, &primitive.conductance)?
            .into_dimensionality::<Ix5>()
            .map_err(|_| LayoutError::schema(&context, "conductance must be five-dimensional"))?;
        let capacitance = read_array(&mut archive, &primitive.capacitance)?
            .into_dimensionality::<Ix5>()
            .map_err(|_| LayoutError::schema(&context, "capacitance must be five-dimensional"))?;
        let expected_shape = [
            axes[0].len(),
            axes[1].len(),
            axes[2].len(),
            primitive.port_order.len(),
            primitive.port_order.len(),
        ];
        if conductance.shape() != expected_shape || capacitance.shape() != expected_shape {
            return Err(LayoutError::schema(
                &context,
                format!("G and C must have shape {expected_shape:?}"),
            ));
        }
        validate_grid(&conductance, &context, "conductance")?;
        validate_grid(&capacitance, &context, "capacitance")?;

        let device_capacitance_correction =
            match (format_version, primitive.device_capacitance_correction) {
                (1, None) => None,
                (1, Some(_)) => {
                    return Err(LayoutError::schema(
                        &context,
                        "version 1 must not contain a device capacitance correction",
                    ));
                }
                (2, None) => {
                    return Err(LayoutError::schema(
                        &context,
                        "version 2 requires a device capacitance correction",
                    ));
                }
                (2, Some(correction)) => Some(read_device_correction(
                    &mut archive,
                    correction,
                    &axes,
                    primitive.port_order.len(),
                    &mut used_paths,
                    &context,
                )?),
                _ => unreachable!("format version was validated"),
            };

        let model = PhysicalPrimitive {
            name: primitive.name.clone(),
            ports: primitive.port_order,
            lengths: axes.remove(0),
            finger_widths: axes.remove(0),
            finger_counts: axes.remove(0),
            conductance,
            capacitance,
            device_capacitance_correction,
        };
        if primitives.insert(primitive.name, model).is_some() {
            return Err(LayoutError::schema(&context, "duplicate primitive name"));
        }
    }
    Ok(PhysicalLookupTable {
        metadata: PhysicalMetadata {
            format_version,
            pdk: manifest.pdk,
            layout_policy: manifest.layout_policy,
            generator: manifest.generator,
        },
        primitives,
    })
}

fn read_device_correction<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    correction: DeviceCorrectionManifest,
    physical_axes: &[Vec<f64>],
    port_count: usize,
    used_paths: &mut BTreeSet<String>,
    context: &str,
) -> Result<DeviceCapacitanceCorrection, LayoutError> {
    if correction.definition != CORRECTION_DEFINITION {
        return Err(LayoutError::schema(
            context,
            format!("device correction definition must be '{CORRECTION_DEFINITION}'"),
        ));
    }
    if correction
        .axis_order
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        != CORRECTION_AXIS_ORDER
    {
        return Err(LayoutError::schema(
            context,
            format!("device correction axis_order must be {CORRECTION_AXIS_ORDER:?}"),
        ));
    }
    if correction.axes.len() != CORRECTION_AXIS_ORDER.len()
        || correction
            .axes
            .iter()
            .map(|axis| axis.name.as_str())
            .ne(CORRECTION_AXIS_ORDER)
    {
        return Err(LayoutError::schema(
            context,
            "device correction axes must follow axis_order",
        ));
    }
    let mut axes = Vec::with_capacity(CORRECTION_AXIS_ORDER.len());
    for member in &correction.axes {
        register_path(&member.path, used_paths, context)?;
        let array = read_array(archive, &member.path)?;
        if array.ndim() != 1 {
            return Err(LayoutError::schema(
                context,
                format!(
                    "device correction axis '{}' must be one-dimensional",
                    member.name
                ),
            ));
        }
        let values = array.into_raw_vec_and_offset().0;
        validate_axis(&values, context, &member.name)?;
        axes.push(values);
    }
    if axes[0] != physical_axes[0] || axes[1] != physical_axes[1] {
        return Err(LayoutError::schema(
            context,
            "device correction length and finger_width axes must match the physical grid",
        ));
    }
    if axes[0].iter().any(|value| *value <= 0.0) || axes[1].iter().any(|value| *value <= 0.0) {
        return Err(LayoutError::schema(
            context,
            "device correction length and finger_width axes must be positive",
        ));
    }
    if axes[2]
        .iter()
        .any(|value| *value < 1.0 || *value > f64::from(u32::MAX) || value.fract() != 0.0)
    {
        return Err(LayoutError::schema(
            context,
            "device correction nf axis must contain positive integers",
        ));
    }
    if axes[2]
        .iter()
        .any(|value| !physical_axes[2].contains(value))
    {
        return Err(LayoutError::schema(
            context,
            "device correction nf axis must be a subset of the physical nf axis",
        ));
    }

    register_path(&correction.capacitance, used_paths, context)?;
    let capacitance = read_array(archive, &correction.capacitance)?;
    let expected_shape = [
        axes[0].len(),
        axes[1].len(),
        axes[2].len(),
        axes[3].len(),
        axes[4].len(),
        axes[5].len(),
        port_count,
        port_count,
    ];
    if capacitance.shape() != expected_shape {
        return Err(LayoutError::schema(
            context,
            format!("device correction capacitance must have shape {expected_shape:?}"),
        ));
    }
    validate_device_correction_grid(&capacitance, context, "device capacitance correction")?;
    Ok(DeviceCapacitanceCorrection {
        finger_counts: axes[2].clone(),
        vbs: axes[3].clone(),
        vgs: axes[4].clone(),
        vds: axes[5].clone(),
        capacitance,
    })
}

fn register_path(
    path: &str,
    used_paths: &mut BTreeSet<String>,
    context: &str,
) -> Result<(), LayoutError> {
    if path.is_empty() || !path.ends_with(".npy") || !used_paths.insert(path.to_owned()) {
        return Err(LayoutError::schema(
            context,
            format!("invalid or duplicate NPY member path '{path}'"),
        ));
    }
    Ok(())
}

fn read_array<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    path: &str,
) -> Result<ArrayD<f64>, LayoutError> {
    let mut member = archive.by_name(path)?;
    let mut bytes = Vec::new();
    member.read_to_end(&mut bytes)?;
    Ok(ArrayD::<f64>::read_npy(Cursor::new(bytes))?)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

    use ndarray::{Array1, Array5, ArrayD, IxDyn};
    use ndarray_npy::WriteNpyExt;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    use crate::{LayoutAwarePoint, PhysicalLookupTable, PhysicalPoint};

    #[test]
    fn opens_the_versioned_npz_archive() {
        let cursor = Cursor::new(Vec::new());
        let mut archive = ZipWriter::new(cursor);
        let options = SimpleFileOptions::default();
        archive.start_file("manifest.json", options).unwrap();
        archive
            .write_all(
                br#"{
                    "format":"shapeic-physical-lut",
                    "version":1,
                    "pdk":"test",
                    "layout_policy":"test",
                    "generator":"unit-test",
                    "primitives":[{
                        "name":"pair",
                        "port_order":["P","N"],
                        "axis_order":["length","finger_width","nf"],
                        "axes":[
                            {"name":"length","path":"p/l.npy"},
                            {"name":"finger_width","path":"p/w.npy"},
                            {"name":"nf","path":"p/n.npy"}
                        ],
                        "conductance":"p/g.npy",
                        "capacitance":"p/c.npy"
                    }]
                }"#,
            )
            .unwrap();
        for (path, values) in [
            ("p/l.npy", vec![0.4e-6, 0.8e-6]),
            ("p/w.npy", vec![0.15e-6, 10.0e-6]),
            ("p/n.npy", vec![1.0, 20.0]),
        ] {
            let mut npy = Vec::new();
            Array1::from(values).write_npy(&mut npy).unwrap();
            archive.start_file(path, options).unwrap();
            archive.write_all(&npy).unwrap();
        }
        let mut matrix = Array5::zeros((2, 2, 2, 2, 2));
        for li in 0..2 {
            for wi in 0..2 {
                for ni in 0..2 {
                    let value = (li + wi + ni + 1) as f64;
                    matrix[(li, wi, ni, 0, 0)] = value;
                    matrix[(li, wi, ni, 0, 1)] = -value;
                    matrix[(li, wi, ni, 1, 0)] = -value;
                    matrix[(li, wi, ni, 1, 1)] = value;
                }
            }
        }
        for path in ["p/g.npy", "p/c.npy"] {
            let mut npy = Vec::new();
            matrix.write_npy(&mut npy).unwrap();
            archive.start_file(path, options).unwrap();
            archive.write_all(&npy).unwrap();
        }
        let bytes = archive.finish().unwrap().into_inner();

        let table = PhysicalLookupTable::from_reader(Cursor::new(bytes)).unwrap();
        assert_eq!(table.metadata().format_version, 1);
        assert_eq!(table.metadata().pdk, "test");
        let result = table
            .primitive("pair")
            .unwrap()
            .query(PhysicalPoint::new(0.6e-6, 5.075e-6, 10))
            .unwrap();
        assert_eq!(result.ports, ["P", "N"]);
        assert!(result.conductance[(0, 0)] > 1.0);
    }

    #[test]
    fn opens_v2_and_interpolates_the_device_correction() {
        let correction = valid_device_correction(&[1.0, 2.0], &[1.0, 3.0], &[1.0, 4.0]);
        let bytes = v2_archive(&[1.0, 2.0], &[1.0, 3.0], &[1.0, 4.0], &correction);
        let table = PhysicalLookupTable::from_reader(Cursor::new(bytes)).unwrap();
        assert_eq!(table.metadata().format_version, 2);
        let result = table
            .primitive("pair")
            .unwrap()
            .query_layout_aware(LayoutAwarePoint::new(1.5, 2.0, 2, -0.5, 1.0, 2.0))
            .unwrap();
        assert_eq!(result.ports, ["P", "N", "B"]);
        assert_ne!(
            result.device_capacitance_correction[(0, 1)],
            result.device_capacitance_correction[(1, 0)]
        );
    }

    #[test]
    fn rejects_v2_geometry_axes_that_do_not_match_the_physical_grid() {
        let correction = valid_device_correction(&[1.0, 2.0], &[1.0, 4.0], &[1.0, 4.0]);
        let bytes = v2_archive(&[1.0, 2.0], &[1.0, 4.0], &[1.0, 4.0], &correction);
        let error = PhysicalLookupTable::from_reader(Cursor::new(bytes)).unwrap_err();
        assert!(error.to_string().contains("must match the physical grid"));
    }

    #[test]
    fn rejects_v2_correction_nf_outside_the_physical_grid() {
        let correction = valid_device_correction(&[1.0, 2.0], &[1.0, 3.0], &[1.0, 3.0]);
        let bytes = v2_archive(&[1.0, 2.0], &[1.0, 3.0], &[1.0, 3.0], &correction);
        let error = PhysicalLookupTable::from_reader(Cursor::new(bytes)).unwrap_err();
        assert!(error.to_string().contains("must be a subset"));
    }

    #[test]
    fn rejects_v2_device_correction_that_does_not_conserve_charge() {
        let mut correction = valid_device_correction(&[1.0, 2.0], &[1.0, 3.0], &[1.0, 4.0]);
        correction[IxDyn(&[0, 0, 0, 0, 0, 0, 0, 0])] += 1.0;
        let bytes = v2_archive(&[1.0, 2.0], &[1.0, 3.0], &[1.0, 4.0], &correction);
        let error = PhysicalLookupTable::from_reader(Cursor::new(bytes)).unwrap_err();
        assert!(error.to_string().contains("does not conserve charge"));
    }

    fn valid_device_correction(
        lengths: &[f64],
        widths: &[f64],
        finger_counts: &[f64],
    ) -> ArrayD<f64> {
        let shape = [
            lengths.len(),
            widths.len(),
            finger_counts.len(),
            2,
            2,
            2,
            3,
            3,
        ];
        ArrayD::from_shape_fn(IxDyn(&shape), |index| {
            let value = (1
                + index[0]
                + 2 * index[1]
                + 4 * index[2]
                + 8 * index[3]
                + 16 * index[4]
                + 32 * index[5]) as f64;
            match (index[6], index[7]) {
                (0, 0) | (1, 1) | (2, 2) => value,
                (0, 1) | (1, 2) | (2, 0) => -value,
                _ => 0.0,
            }
        })
    }

    fn v2_archive(
        correction_lengths: &[f64],
        correction_widths: &[f64],
        correction_finger_counts: &[f64],
        correction: &ArrayD<f64>,
    ) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut archive = ZipWriter::new(cursor);
        let options = SimpleFileOptions::default();
        archive.start_file("manifest.json", options).unwrap();
        archive
            .write_all(
                br#"{
                    "format":"shapeic-physical-lut",
                    "version":2,
                    "pdk":"test",
                    "layout_policy":"test",
                    "generator":"unit-test",
                    "primitives":[{
                        "name":"pair",
                        "port_order":["P","N","B"],
                        "axis_order":["length","finger_width","nf"],
                        "axes":[
                            {"name":"length","path":"p/l.npy"},
                            {"name":"finger_width","path":"p/w.npy"},
                            {"name":"nf","path":"p/n.npy"}
                        ],
                        "conductance":"p/g.npy",
                        "capacitance":"p/c.npy",
                        "device_capacitance_correction":{
                            "definition":"pex_mos_only_minus_aggregate_compact_model",
                            "axis_order":["length","finger_width","nf","vbs","vgs","vds"],
                            "axes":[
                                {"name":"length","path":"p/dc/l.npy"},
                                {"name":"finger_width","path":"p/dc/w.npy"},
                                {"name":"nf","path":"p/dc/n.npy"},
                                {"name":"vbs","path":"p/dc/vbs.npy"},
                                {"name":"vgs","path":"p/dc/vgs.npy"},
                                {"name":"vds","path":"p/dc/vds.npy"}
                            ],
                            "capacitance":"p/dc/c.npy"
                        }
                    }]
                }"#,
            )
            .unwrap();
        for (path, values) in [
            ("p/l.npy", &[1.0, 2.0][..]),
            ("p/w.npy", &[1.0, 3.0][..]),
            ("p/n.npy", &[1.0, 2.0, 4.0][..]),
            ("p/dc/l.npy", correction_lengths),
            ("p/dc/w.npy", correction_widths),
            ("p/dc/n.npy", correction_finger_counts),
            ("p/dc/vbs.npy", &[-1.0, 0.0]),
            ("p/dc/vgs.npy", &[0.0, 2.0]),
            ("p/dc/vds.npy", &[0.0, 4.0]),
        ] {
            let mut npy = Vec::new();
            Array1::from(values.to_vec()).write_npy(&mut npy).unwrap();
            archive.start_file(path, options).unwrap();
            archive.write_all(&npy).unwrap();
        }
        let matrix = Array5::<f64>::zeros((2, 2, 3, 3, 3));
        for path in ["p/g.npy", "p/c.npy"] {
            let mut npy = Vec::new();
            matrix.write_npy(&mut npy).unwrap();
            archive.start_file(path, options).unwrap();
            archive.write_all(&npy).unwrap();
        }
        let mut npy = Vec::new();
        correction.write_npy(&mut npy).unwrap();
        archive.start_file("p/dc/c.npy", options).unwrap();
        archive.write_all(&npy).unwrap();
        archive.finish().unwrap().into_inner()
    }
}
