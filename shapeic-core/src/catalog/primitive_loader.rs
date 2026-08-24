use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::primitive_catalog::PrimitiveCatalog;

use crate::primitive::build::PrimitiveBuildSpec;
use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};
use crate::primitive::manifest::PrimitivePhysicalModel;
use crate::primitive::small_signal::SmallSignalModel;

#[derive(Debug)]
pub enum PrimitiveLoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
    ExternalBuildJson {
        path: std::path::PathBuf,
        source: serde_json::Error,
    },
    MissingPort {
        primitive: String,
        pin: String,
    },
    MissingNetlistFile {
        primitive: String,
    },
}

impl From<std::io::Error> for PrimitiveLoadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<serde_json::Error> for PrimitiveLoadError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug, Deserialize)]
struct RawPort {
    name: Option<String>,
    role: PinRole,
}
#[derive(Debug, Deserialize)]
struct RawPrimitiveFiles {
    netlist: Option<String>,
    build: Option<String>,
    symbol: Option<String>,
}

pub fn load_primitive_manifest(path: &Path) -> Result<PrimitiveManifest, PrimitiveLoadError> {
    let content = fs::read_to_string(path)?;
    let raw: RawPrimitiveManifest = serde_json::from_str(&content)?;
    raw.into_manifest(path.parent().unwrap_or_else(|| Path::new(".")))
}

pub fn load_primitive_catalog(
    primitives_dir: &Path,
) -> Result<PrimitiveCatalog, PrimitiveLoadError> {
    let mut catalog = PrimitiveCatalog::new();

    for entry in fs::read_dir(primitives_dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let manifest_path = path.join("primitive.json");
        if !manifest_path.exists() {
            continue;
        }

        catalog.register(load_primitive_manifest(&manifest_path)?);
    }

    Ok(catalog)
}

#[derive(Debug, Deserialize)]
struct RawPrimitiveManifest {
    name: String,
    version: Option<String>,
    description: Option<String>,
    subckt_name: Option<String>,
    transistor_type: Option<String>,
    pin_order: Option<Vec<String>>,
    ports: HashMap<String, RawPort>,
    layout_params: Option<serde_json::Value>,
    lut_config: Option<serde_json::Value>,
    files: RawPrimitiveFiles,
    small_signal: Option<SmallSignalModel>,
    physical_model: Option<PrimitivePhysicalModel>,
    build: Option<PrimitiveBuildSpec>,
}

impl RawPrimitiveManifest {
    fn into_manifest(self, base_dir: &Path) -> Result<PrimitiveManifest, PrimitiveLoadError> {
        let pin_order = self
            .pin_order
            .unwrap_or_else(|| sorted_port_names(&self.ports));

        let mut pins = Vec::with_capacity(pin_order.len());
        for pin_name in pin_order {
            let port =
                self.ports
                    .get(&pin_name)
                    .ok_or_else(|| PrimitiveLoadError::MissingPort {
                        primitive: self.name.clone(),
                        pin: pin_name.clone(),
                    })?;

            pins.push(Pin {
                name: port.name.clone().unwrap_or(pin_name),
                role: port.role.clone(),
            });
        }

        let netlist = self
            .files
            .netlist
            .ok_or_else(|| PrimitiveLoadError::MissingNetlistFile {
                primitive: self.name.clone(),
            })?;

        let files = PrimitiveFiles {
            netlist,
            build: self.files.build.clone(),
            symbol: self.files.symbol.clone(),
        };
        let build = match self.build {
            Some(build) => Some(build),
            None => load_external_build_spec(base_dir, files.build.as_deref())?,
        };

        Ok(PrimitiveManifest {
            subckt_name: self.subckt_name.unwrap_or_else(|| self.name.clone()),
            name: self.name,
            version: self.version.unwrap_or_else(|| "0.1.0".to_string()),
            description: self.description,
            pins,
            files,
            small_signal: self.small_signal,
            physical_model: self.physical_model,
            transistor_type: self.transistor_type,
            layout_params: self.layout_params,
            lut_config: self.lut_config,
            build,
        })
    }
}

fn sorted_port_names(ports: &HashMap<String, RawPort>) -> Vec<String> {
    let mut names = ports.keys().cloned().collect::<Vec<_>>();
    names.sort();
    names
}

fn load_external_build_spec(
    base_dir: &Path,
    build_file: Option<&str>,
) -> Result<Option<PrimitiveBuildSpec>, PrimitiveLoadError> {
    let Some(build_file) = build_file else {
        return Ok(None);
    };

    if !build_file.ends_with(".json") {
        return Ok(None);
    }

    let path = base_dir.join(build_file);
    let content = fs::read_to_string(&path)?;
    serde_json::from_str(&content)
        .map(Some)
        .map_err(|source| PrimitiveLoadError::ExternalBuildJson { path, source })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::load_primitive_manifest;

    #[test]
    fn loads_the_existing_small_signal_manifest_topology() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir
            .parent()
            .unwrap()
            .join("analoglib/primitives/simplediffpair/primitive.json");

        let primitive = load_primitive_manifest(&path).unwrap();
        let model = primitive.small_signal.as_ref().unwrap();
        let physical = primitive.physical_model.as_ref().unwrap();

        assert_eq!(model.branches().len(), 2);
        let m2 = model.branch("m2").unwrap();
        assert_eq!(m2.drain_pin(), "VOUTN");
        assert_eq!(m2.gate_pin(), "VINN");
        assert_eq!(m2.source_pin(), "VTAIL");
        assert_eq!(m2.bulk_pin(), "VSS");
        assert_eq!(physical.lut_primitive(), "simplediffpair");
        assert_eq!(physical.operating_point_branch(), "m1");
        assert_eq!(physical.ports().len(), 6);
        assert_eq!(
            physical
                .ports()
                .iter()
                .find(|mapping| mapping.physical_port() == "B")
                .map(|mapping| mapping.primitive_pin()),
            Some("VSS")
        );
    }

    #[test]
    fn loads_the_current_mirror_physical_port_mapping() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir
            .parent()
            .unwrap()
            .join("analoglib/primitives/simplecurrentmirror/primitive.json");

        let primitive = load_primitive_manifest(&path).unwrap();
        let physical = primitive.physical_model.as_ref().unwrap();

        assert_eq!(physical.lut_primitive(), "currentmirror");
        assert_eq!(physical.operating_point_branch(), "m1");
        assert_eq!(
            physical
                .ports()
                .iter()
                .map(|mapping| (mapping.physical_port(), mapping.primitive_pin()))
                .collect::<Vec<_>>(),
            [
                ("DOUT", "VOUTP"),
                ("DREF", "VINP"),
                ("S", "VDD"),
                ("B", "VDD"),
            ]
        );
    }
}
