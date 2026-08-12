use std::collections::HashMap;

use crate::primitive::manifest::PrimitiveManifest;

#[derive(Debug, Clone, Default)]
pub struct PrimitiveCatalog {
    primitives: HashMap<String, PrimitiveManifest>,
}

impl PrimitiveCatalog {
    pub fn new() -> Self {
        Self {
            primitives: HashMap::new(),
        }
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
}
