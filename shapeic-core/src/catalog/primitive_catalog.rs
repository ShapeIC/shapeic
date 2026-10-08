use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use shapeic_lut::CurrentSizingLimits;

use crate::primitive::manifest::PrimitiveManifest;

#[derive(Debug, Clone, Default)]
pub struct PrimitiveCatalog {
    primitives: HashMap<String, PrimitiveManifest>,
    primitives_dir: Option<PathBuf>,
}

impl PrimitiveCatalog {
    pub fn new() -> Self {
        Self {
            primitives: HashMap::new(),
            primitives_dir: None,
        }
    }

    pub(crate) fn from_directory(path: &Path) -> Self {
        Self {
            primitives_dir: Some(path.to_path_buf()),
            ..Self::new()
        }
    }

    pub fn has_cellkit_source(&self) -> bool {
        self.primitives_dir.is_some()
    }

    /// Read the selected PDK's optional PCell limits without loading its provider.
    pub fn geometry_limits(
        &self,
        primitive: &str,
        pdk: &str,
    ) -> Result<Option<CurrentSizingLimits>, String> {
        let root = self
            .primitives_dir
            .as_ref()
            .ok_or("catalog has no CellKit source")?;
        if pdk.is_empty() || pdk == "." || pdk == ".." || pdk.contains(['/', '\\']) {
            return Err(format!("invalid PDK name '{pdk}'"));
        }
        if primitive.is_empty()
            || primitive == "."
            || primitive == ".."
            || primitive.contains(['/', '\\'])
        {
            return Err(format!("invalid primitive name '{primitive}'"));
        }
        let path = root.join(primitive).join(pdk).join("geometry.json");
        let raw = match fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("{}: {error}", path.display())),
        };
        let fields: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&raw).map_err(|error| format!("{}: {error}", path.display()))?;
        if fields.values().any(serde_json::Value::is_null) {
            return Err(format!(
                "{}: geometry limits cannot be null",
                path.display()
            ));
        }
        let limits: GeometryLimitsFile = serde_json::from_value(serde_json::Value::Object(fields))
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if limits.required_nf.is_none()
            && limits.max_finger_width_m.is_none()
            && limits.nf_multiple_of.is_none()
        {
            return Err(format!(
                "{}: at least one geometry limit is required",
                path.display()
            ));
        }
        if limits.required_nf == Some(0)
            || limits.nf_multiple_of == Some(0)
            || limits
                .max_finger_width_m
                .is_some_and(|width| !width.is_finite() || width <= 0.0)
        {
            return Err(format!(
                "{}: geometry limits must be positive and finite",
                path.display()
            ));
        }
        Ok(Some(CurrentSizingLimits {
            required_nf: limits.required_nf,
            max_finger_width_m: limits.max_finger_width_m,
            nf_multiple_of: limits.nf_multiple_of,
        }))
    }

    pub fn register(&mut self, primitive: PrimitiveManifest) {
        self.primitives.insert(primitive.name.clone(), primitive);
    }

    pub fn get(&self, name: &str) -> Option<&PrimitiveManifest> {
        self.primitives.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.primitives.contains_key(name)
    }

    pub fn list(&self) -> Vec<&PrimitiveManifest> {
        let mut primitives = self.primitives.values().collect::<Vec<_>>();
        primitives.sort_by(|left, right| left.name.cmp(&right.name));
        primitives
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GeometryLimitsFile {
    required_nf: Option<u32>,
    max_finger_width_m: Option<f64>,
    nf_multiple_of: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles};

    #[test]
    fn registers_and_lists_primitives() {
        let mut catalog = PrimitiveCatalog::new();

        catalog.register(PrimitiveManifest {
            name: "simplediffpair".to_string(),
            version: "1.0".to_string(),
            description: None,
            subckt_name: "simplediffpair".to_string(),
            pins: vec![
                Pin {
                    name: "VINP".to_string(),
                    role: PinRole::Input,
                },
                Pin {
                    name: "VOUTP".to_string(),
                    role: PinRole::Output,
                },
            ],
            files: PrimitiveFiles {
                netlist: "netlist/netlist.spice".to_string(),
                build: None,
                symbol: None,
            },
            small_signal: None,
            physical_model: None,
            transistor_type: None,
            layout_params: None,
            lut_config: None,
            build: None,
        });

        assert!(catalog.contains("simplediffpair"));
        assert_eq!(
            catalog
                .get("simplediffpair")
                .map(|primitive| primitive.subckt_name.as_str()),
            Some("simplediffpair")
        );
        assert_eq!(catalog.list().len(), 1);
    }

    #[test]
    fn reads_pdk_geometry_limits_from_loaded_cellkit() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../shapeic-cellkit/primitives");
        let catalog = crate::catalog::primitive_loader::load_primitive_catalog(&root).unwrap();
        let limits = catalog
            .geometry_limits("simplediffpair", "ihp-sg13g2")
            .unwrap()
            .unwrap();
        assert_eq!(limits.required_nf, Some(4));
        assert_eq!(limits.max_finger_width_m, Some(1e-5));
        assert_eq!(limits.nf_multiple_of, None);
        let mirror_limits = catalog
            .geometry_limits("simplecurrentmirror", "ihp-sg13g2")
            .unwrap()
            .unwrap();
        assert_eq!(mirror_limits.required_nf, None);
        assert_eq!(mirror_limits.max_finger_width_m, Some(1e-5));
        assert_eq!(mirror_limits.nf_multiple_of, Some(2));
        assert_eq!(
            catalog
                .geometry_limits("simplediffpair", "sky130A")
                .unwrap(),
            None
        );
        assert!(
            PrimitiveCatalog::new()
                .geometry_limits("simplediffpair", "ihp-sg13g2")
                .is_err()
        );
    }

    #[test]
    fn rejects_invalid_geometry_limits() {
        let root = std::env::temp_dir().join(format!(
            "shapeic-geometry-limits-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let file = root.join("pair/test/geometry.json");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        let catalog = PrimitiveCatalog::from_directory(&root);
        for invalid in [
            r#"{"required_nf":0}"#,
            r#"{"max_finger_width_m":-1}"#,
            r#"{"nf_multiple_of":0}"#,
            r#"{"nf_multiple_of":-2}"#,
            r#"{"nf_multiple_of":2.0}"#,
            r#"{"nf_multiple_of":true}"#,
            r#"{"required_nf":null,"max_finger_width_m":1e-6}"#,
            r#"{"unknown":1}"#,
        ] {
            fs::write(&file, invalid).unwrap();
            assert!(catalog.geometry_limits("pair", "test").is_err(), "{invalid}");
        }
        fs::remove_dir_all(root).unwrap();
    }
}
