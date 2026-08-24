use std::collections::HashSet;
use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::build::PrimitiveBuildSpec;
use super::small_signal::SmallSignalModel;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveManifest {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub subckt_name: String,
    pub pins: Vec<Pin>,
    pub files: PrimitiveFiles,
    pub small_signal: Option<SmallSignalModel>,
    pub physical_model: Option<PrimitivePhysicalModel>,
    pub transistor_type: Option<String>,
    pub layout_params: Option<serde_json::Value>,
    pub lut_config: Option<serde_json::Value>,
    pub build: Option<PrimitiveBuildSpec>,
}

/// Physical LUT identity, operating-point branch, and logical port mapping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrimitivePhysicalModel {
    lut_primitive: String,
    operating_point_branch: String,
    #[serde(with = "physical_port_map")]
    ports: Vec<PhysicalPortMapping>,
}

impl PrimitivePhysicalModel {
    /// Creates a typed physical descriptor from logical LUT ports and primitive pins.
    pub fn new(
        lut_primitive: impl Into<String>,
        operating_point_branch: impl Into<String>,
        ports: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            lut_primitive: lut_primitive.into(),
            operating_point_branch: operating_point_branch.into(),
            ports: ports
                .into_iter()
                .map(|(physical_port, primitive_pin)| {
                    PhysicalPortMapping::new(physical_port, primitive_pin)
                })
                .collect(),
        }
    }

    /// Returns the primitive key used by the physical LUT.
    pub fn lut_primitive(&self) -> &str {
        &self.lut_primitive
    }

    /// Returns the LUT branch whose candidate columns define the physical query point.
    pub fn operating_point_branch(&self) -> &str {
        &self.operating_point_branch
    }

    /// Returns physical LUT port mappings in manifest declaration order.
    pub fn ports(&self) -> &[PhysicalPortMapping] {
        &self.ports
    }
}

/// One logical physical LUT port mapped to a primitive circuit pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPortMapping {
    physical_port: String,
    primitive_pin: String,
}

impl PhysicalPortMapping {
    /// Creates one physical-port-to-primitive-pin mapping.
    pub fn new(physical_port: impl Into<String>, primitive_pin: impl Into<String>) -> Self {
        Self {
            physical_port: physical_port.into(),
            primitive_pin: primitive_pin.into(),
        }
    }

    /// Returns the logical port name stored in the physical LUT.
    pub fn physical_port(&self) -> &str {
        &self.physical_port
    }

    /// Returns the primitive manifest pin connected to this physical port.
    pub fn primitive_pin(&self) -> &str {
        &self.primitive_pin
    }
}

mod physical_port_map {
    use super::*;

    pub fn serialize<S>(ports: &[PhysicalPortMapping], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(ports.len()))?;
        for port in ports {
            map.serialize_entry(port.physical_port(), port.primitive_pin())?;
        }
        map.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<PhysicalPortMapping>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PortMapVisitor;

        impl<'de> Visitor<'de> for PortMapVisitor {
            type Value = Vec<PhysicalPortMapping>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a map from physical LUT ports to primitive pins")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut ports = Vec::with_capacity(map.size_hint().unwrap_or(0));
                let mut names = HashSet::new();
                while let Some((physical_port, primitive_pin)) =
                    map.next_entry::<String, String>()?
                {
                    if !names.insert(physical_port.clone()) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate physical port '{physical_port}'"
                        )));
                    }
                    ports.push(PhysicalPortMapping::new(physical_port, primitive_pin));
                }
                Ok(ports)
            }
        }

        deserializer.deserialize_map(PortMapVisitor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pin {
    pub name: String,
    pub role: PinRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PinRole {
    Input,
    Output,
    Bias,
    Supply,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrimitiveFiles {
    pub netlist: String,
    pub build: Option<String>,
    pub symbol: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::PrimitivePhysicalModel;

    #[test]
    fn physical_model_uses_the_manifest_port_map_shape() {
        let model: PrimitivePhysicalModel = serde_json::from_str(
            r#"{
                "lut_primitive":"pair",
                "operating_point_branch":"m1",
                "ports":{"D":"VOUT","G":"VIN","S":"VSS","B":"VSS"}
            }"#,
        )
        .unwrap();

        assert_eq!(model.lut_primitive(), "pair");
        assert_eq!(model.operating_point_branch(), "m1");
        assert_eq!(model.ports()[0].physical_port(), "D");
        assert_eq!(model.ports()[0].primitive_pin(), "VOUT");
        assert_eq!(serde_json::to_value(model).unwrap()["ports"]["G"], "VIN");
    }

    #[test]
    fn rejects_duplicate_physical_ports_while_deserializing() {
        let error = serde_json::from_str::<PrimitivePhysicalModel>(
            r#"{
                "lut_primitive":"pair",
                "operating_point_branch":"m1",
                "ports":{"D":"VOUT","D":"OTHER"}
            }"#,
        )
        .expect_err("duplicate physical ports must not be silently overwritten");

        assert!(error.to_string().contains("duplicate physical port 'D'"));
    }
}
