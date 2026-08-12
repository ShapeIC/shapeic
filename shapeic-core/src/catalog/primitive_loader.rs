use std::collections::HashMap;
use std::path::Path;
use std::fs;
use serde::Deserialize;

use super::primitive_catalog::PrimitiveCatalog;

use crate::primitive::build::PrimitiveBuildSpec;
use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};

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
