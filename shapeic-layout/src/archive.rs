use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Cursor, Read, Seek};
use std::path::Path;

use ndarray::{ArrayD, Ix5};
use ndarray_npy::ReadNpyExt;
use serde::Deserialize;
use zip::ZipArchive;

use crate::LayoutError;
use crate::model::{PhysicalLookupTable, PhysicalMetadata, PhysicalPrimitive};
use crate::validation::{validate_axis, validate_grid};

const MANIFEST_MEMBER: &str = "manifest.json";
const FORMAT: &str = "shapeic-physical-lut";
const VERSION: u32 = 1;
const AXIS_ORDER: [&str; 3] = ["length", "finger_width", "nf"];

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
    if manifest.format != FORMAT || manifest.version != VERSION {
        return Err(LayoutError::schema(
            MANIFEST_MEMBER,
            format!(
                "expected format '{FORMAT}' version {VERSION}, found '{}' version {}",
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

        let model = PhysicalPrimitive {
            name: primitive.name.clone(),
            ports: primitive.port_order,
            lengths: axes.remove(0),
            finger_widths: axes.remove(0),
            finger_counts: axes.remove(0),
            conductance,
            capacitance,
        };
        if primitives.insert(primitive.name, model).is_some() {
            return Err(LayoutError::schema(&context, "duplicate primitive name"));
        }
    }
    Ok(PhysicalLookupTable {
        metadata: PhysicalMetadata {
            pdk: manifest.pdk,
            layout_policy: manifest.layout_policy,
            generator: manifest.generator,
        },
        primitives,
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

    use ndarray::{Array1, Array5};
    use ndarray_npy::WriteNpyExt;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    use crate::{PhysicalLookupTable, PhysicalPoint};

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
        assert_eq!(table.metadata().pdk, "test");
        let result = table
            .primitive("pair")
            .unwrap()
            .query(PhysicalPoint::new(0.6e-6, 5.075e-6, 10))
            .unwrap();
        assert_eq!(result.ports, ["P", "N"]);
        assert!(result.conductance[(0, 0)] > 1.0);
    }
}
