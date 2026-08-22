use serde::{Deserialize, Serialize};

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
    pub transistor_type: Option<String>,
    pub layout_params: Option<serde_json::Value>,
    pub lut_config: Option<serde_json::Value>,
    pub build: Option<PrimitiveBuildSpec>,
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
