//! Small-signal topology published by primitive definitions.

use serde::{Deserialize, Serialize};

/// Small-signal topology of a primitive, split into MOS-like branches.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmallSignalModel {
    branches: Vec<SmallSignalBranch>,
}

impl SmallSignalModel {
    /// Creates a model from its branches in declaration order.
    pub fn new(branches: Vec<SmallSignalBranch>) -> Self {
        Self { branches }
    }

    /// Returns the small-signal branches in declaration order.
    pub fn branches(&self) -> &[SmallSignalBranch] {
        &self.branches
    }

    /// Finds a branch by its local name.
    pub fn branch(&self, name: &str) -> Option<&SmallSignalBranch> {
        self.branches.iter().find(|branch| branch.name == name)
    }
}

/// Pin mapping for one MOS-like small-signal branch.
///
/// The branch identifies topology only. Candidate-dependent values such as
/// `gm`, `ro`, and capacitances retain the existing `<parameter>__<branch>`
/// build-column convention.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmallSignalBranch {
    name: String,
    #[serde(rename = "vd")]
    drain_pin: String,
    #[serde(rename = "vg")]
    gate_pin: String,
    #[serde(rename = "vs")]
    source_pin: String,
    #[serde(rename = "vb")]
    bulk_pin: String,
}

impl SmallSignalBranch {
    /// Creates one branch from its name and drain, gate, source, and bulk pins.
    pub fn new(
        name: impl Into<String>,
        drain_pin: impl Into<String>,
        gate_pin: impl Into<String>,
        source_pin: impl Into<String>,
        bulk_pin: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            drain_pin: drain_pin.into(),
            gate_pin: gate_pin.into(),
            source_pin: source_pin.into(),
            bulk_pin: bulk_pin.into(),
        }
    }

    /// Returns the branch-local name used in candidate parameter names.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the primitive pin connected to the branch drain.
    pub fn drain_pin(&self) -> &str {
        &self.drain_pin
    }

    /// Returns the primitive pin connected to the branch gate.
    pub fn gate_pin(&self) -> &str {
        &self.gate_pin
    }

    /// Returns the primitive pin connected to the branch source.
    pub fn source_pin(&self) -> &str {
        &self.source_pin
    }

    /// Returns the primitive pin connected to the branch bulk.
    pub fn bulk_pin(&self) -> &str {
        &self.bulk_pin
    }
}

#[cfg(test)]
mod tests {
    use super::{SmallSignalBranch, SmallSignalModel};

    #[test]
    fn preserves_branch_order_and_resolves_names() {
        let model = SmallSignalModel::new(vec![
            SmallSignalBranch::new("m1", "VOUTP", "VINP", "VTAIL", "VSS"),
            SmallSignalBranch::new("m2", "VOUTN", "VINN", "VTAIL", "VSS"),
        ]);

        assert_eq!(model.branches().len(), 2);
        assert_eq!(
            model.branch("m2").map(|branch| branch.gate_pin()),
            Some("VINN")
        );
        assert!(model.branch("missing").is_none());
    }

    #[test]
    fn keeps_the_manifest_terminal_field_names() {
        let model: SmallSignalModel = serde_json::from_str(
            r#"{
                "branches": [
                    { "name": "m1", "vd": "VOUT", "vg": "VIN", "vs": "VTAIL", "vb": "VSS" }
                ]
            }"#,
        )
        .unwrap();
        let branch = model.branch("m1").unwrap();

        assert_eq!(branch.drain_pin(), "VOUT");
        assert_eq!(branch.gate_pin(), "VIN");
        assert_eq!(branch.source_pin(), "VTAIL");
        assert_eq!(branch.bulk_pin(), "VSS");
        assert_eq!(
            serde_json::to_value(model).unwrap()["branches"][0]["vd"],
            "VOUT"
        );
    }
}
