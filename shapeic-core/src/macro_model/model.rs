use std::path::{Path, PathBuf};

use crate::analysis::AcMetric;
use crate::circuit::Circuit;
use crate::testbench::AcAnalysis;

/// A reusable analog macro with an implementation, compact model, and
/// exploration definition.
#[derive(Clone, Debug, PartialEq)]
pub struct Macro {
    name: String,
    ports: Vec<MacroPort>,
    circuit: Circuit,
    compact_model: Circuit,
    exploration: MacroExploration,
}

impl Macro {
    /// Creates a macro with an empty exploration definition.
    pub fn new(
        name: impl Into<String>,
        ports: Vec<MacroPort>,
        circuit: Circuit,
        compact_model: Circuit,
    ) -> Self {
        Self {
            name: name.into(),
            ports,
            circuit,
            compact_model,
            exploration: MacroExploration::default(),
        }
    }

    /// Attaches one unprepared AC testbench to the macro.
    pub fn with_ac_testbench(mut self, testbench: MacroAcTestbench) -> Self {
        self.exploration.testbenches.push(testbench);
        self
    }

    /// Exposes one accepted result value as a compact-model parameter.
    pub fn with_compact_output(mut self, binding: MacroCompactOutputBinding) -> Self {
        self.exploration.compact_outputs.push(binding);
        self
    }

    /// Exposes one accepted result value as a public interface variable.
    pub fn with_interface_binding(mut self, binding: MacroInterfaceBinding) -> Self {
        self.exploration.interface_bindings.push(binding);
        self
    }

    /// Replaces the macro's exploration definition.
    pub fn with_exploration(mut self, exploration: MacroExploration) -> Self {
        self.exploration = exploration;
        self
    }

    /// Returns the macro name used by catalogs and submacro references.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the ordered public ports.
    pub fn ports(&self) -> &[MacroPort] {
        &self.ports
    }

    /// Returns the internal circuit explored when this macro is the DUT.
    pub const fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// Returns the compact circuit used when this macro is instantiated by a
    /// parent macro.
    pub const fn compact_model(&self) -> &Circuit {
        &self.compact_model
    }

    /// Returns the testbenches and future exploration configuration owned by
    /// this macro.
    pub const fn exploration(&self) -> &MacroExploration {
        &self.exploration
    }
}

/// One ordered public port of a macro.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroPort {
    name: String,
    role: MacroPortRole,
}

impl MacroPort {
    /// Creates one named macro port with its electrical role.
    pub fn new(name: impl Into<String>, role: MacroPortRole) -> Self {
        Self {
            name: name.into(),
            role,
        }
    }

    /// Returns the public port name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the electrical role of the port.
    pub const fn role(&self) -> MacroPortRole {
        self.role
    }
}

/// Electrical role assigned to a macro port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroPortRole {
    Input,
    Output,
    Inout,
    Bias,
    Supply,
    Ground,
}

/// Analysis definitions owned by one macro.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MacroExploration {
    testbenches: Vec<MacroAcTestbench>,
    compact_outputs: Vec<MacroCompactOutputBinding>,
    interface_bindings: Vec<MacroInterfaceBinding>,
}

impl MacroExploration {
    /// Creates an empty exploration definition.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one unprepared AC testbench.
    pub fn with_ac_testbench(mut self, testbench: MacroAcTestbench) -> Self {
        self.testbenches.push(testbench);
        self
    }

    /// Adds an explicit compact-model parameter binding.
    pub fn with_compact_output(mut self, binding: MacroCompactOutputBinding) -> Self {
        self.compact_outputs.push(binding);
        self
    }

    /// Adds an explicit public interface-variable binding.
    pub fn with_interface_binding(mut self, binding: MacroInterfaceBinding) -> Self {
        self.interface_bindings.push(binding);
        self
    }

    /// Returns the AC testbenches in evaluation order.
    pub fn testbenches(&self) -> &[MacroAcTestbench] {
        &self.testbenches
    }

    /// Finds an AC testbench by name.
    pub fn testbench(&self, name: &str) -> Option<&MacroAcTestbench> {
        self.testbenches
            .iter()
            .find(|testbench| testbench.name == name)
    }

    /// Returns compact-model output bindings in projected column order.
    pub fn compact_outputs(&self) -> &[MacroCompactOutputBinding] {
        &self.compact_outputs
    }

    /// Returns public interface bindings in projected column order.
    pub fn interface_bindings(&self) -> &[MacroInterfaceBinding] {
        &self.interface_bindings
    }
}

/// Explicit source of one value projected from an accepted macro candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroOutputSource {
    /// One exact column from the selected candidate of a circuit instance.
    CandidateColumn {
        instance_path: String,
        column: String,
    },
    /// One metric produced by a named AC testbench.
    AcMetric { testbench: String, metric: AcMetric },
}

impl MacroOutputSource {
    /// Selects one exact candidate column from an implementation instance.
    pub fn candidate_column(instance_path: impl Into<String>, column: impl Into<String>) -> Self {
        Self::CandidateColumn {
            instance_path: instance_path.into(),
            column: column.into(),
        }
    }

    /// Selects one AC metric from a macro-local testbench.
    pub fn ac_metric(testbench: impl Into<String>, metric: AcMetric) -> Self {
        Self::AcMetric {
            testbench: testbench.into(),
            metric,
        }
    }
}

/// Maps one accepted result value to a symbolic compact-model parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroCompactOutputBinding {
    parameter: String,
    source: MacroOutputSource,
}

impl MacroCompactOutputBinding {
    /// Creates one compact-model parameter binding.
    pub fn new(parameter: impl Into<String>, source: MacroOutputSource) -> Self {
        Self {
            parameter: parameter.into(),
            source,
        }
    }

    /// Returns the unscoped compact-model parameter name.
    pub fn parameter(&self) -> &str {
        &self.parameter
    }

    /// Returns the accepted-result source.
    pub const fn source(&self) -> &MacroOutputSource {
        &self.source
    }
}

/// Maps one accepted result value to a public macro port variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroInterfaceBinding {
    port: String,
    source: MacroOutputSource,
}

impl MacroInterfaceBinding {
    /// Creates one public interface-variable binding.
    pub fn new(port: impl Into<String>, source: MacroOutputSource) -> Self {
        Self {
            port: port.into(),
            source,
        }
    }

    /// Returns the public macro port represented by this value.
    pub fn port(&self) -> &str {
        &self.port
    }

    /// Returns the accepted-result source.
    pub const fn source(&self) -> &MacroOutputSource {
        &self.source
    }
}

/// An unprepared AC testbench associated with a macro.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroAcTestbench {
    name: String,
    source: MacroTestbenchSource,
    analysis: AcAnalysis,
    domain: MacroAnalysisDomain,
}

/// Physical modeling domain used by one macro analysis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MacroAnalysisDomain {
    /// Uses only the electrical small-signal and MOS capacitance models.
    #[default]
    Electrical,
    /// Adds interconnect admittance and device-capacitance corrections from a
    /// physical LUT.
    LayoutAware,
}

impl MacroAcTestbench {
    /// Creates a testbench from inline SPICE source text.
    pub fn from_spice(
        name: impl Into<String>,
        source: impl Into<String>,
        analysis: AcAnalysis,
    ) -> Self {
        Self {
            name: name.into(),
            source: MacroTestbenchSource::Spice(source.into()),
            analysis,
            domain: MacroAnalysisDomain::Electrical,
        }
    }

    /// Creates a testbench whose SPICE source will be read from a file during
    /// preparation.
    pub fn from_spice_file(
        name: impl Into<String>,
        path: impl Into<PathBuf>,
        analysis: AcAnalysis,
    ) -> Self {
        Self {
            name: name.into(),
            source: MacroTestbenchSource::SpiceFile(path.into()),
            analysis,
            domain: MacroAnalysisDomain::Electrical,
        }
    }

    /// Selects the physical modeling domain for this testbench.
    pub fn with_domain(mut self, domain: MacroAnalysisDomain) -> Self {
        self.domain = domain;
        self
    }

    /// Returns the testbench name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the unprepared SPICE source location.
    pub const fn source(&self) -> &MacroTestbenchSource {
        &self.source
    }

    /// Returns the attached AC analysis definition.
    pub const fn analysis(&self) -> &AcAnalysis {
        &self.analysis
    }

    /// Returns the physical modeling domain used by this testbench.
    pub const fn domain(&self) -> MacroAnalysisDomain {
        self.domain
    }
}

/// Source of an unprepared macro testbench.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroTestbenchSource {
    /// SPICE source held directly in memory.
    Spice(String),
    /// Path to a `.spice` file read later during preparation.
    SpiceFile(PathBuf),
}

impl MacroTestbenchSource {
    /// Returns the inline source text when this is an in-memory testbench.
    pub fn as_spice(&self) -> Option<&str> {
        match self {
            Self::Spice(source) => Some(source),
            Self::SpiceFile(_) => None,
        }
    }

    /// Returns the file path when this is a file-backed testbench.
    pub fn as_path(&self) -> Option<&Path> {
        match self {
            Self::Spice(_) => None,
            Self::SpiceFile(path) => Some(path),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::analysis::{
        AcMetricSet, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets,
    };
    use crate::circuit::Circuit;
    use crate::testbench::{AcAnalysis, TransferFunction};

    use super::{
        Macro, MacroAcTestbench, MacroAnalysisDomain, MacroExploration, MacroPort, MacroPortRole,
    };

    fn analysis() -> AcAnalysis {
        AcAnalysis::new(
            TransferFunction::new("VIN", "VOUT"),
            AdaptiveAcConfig {
                min_frequency_hz: 1.0,
                max_frequency_hz: 1.0e9,
                coarse_points_per_decade: 4,
                crossing_relative_tolerance: 0.005,
                max_refinement_steps: 32,
                retain_samples: false,
            },
            AdaptiveAcPolicy {
                mode: AnalysisMode::Prune,
                targets: AnalysisTargets::NONE,
                metrics: AcMetricSet::ALL,
            },
        )
    }

    fn gain_stage() -> Macro {
        let circuit = Circuit::builder()
            .primitive(
                "xdp",
                "simplediffpair",
                [("VINP", "VIN"), ("VOUTP", "VOUT")],
            )
            .build();
        let compact_model = Circuit::builder()
            .vccs("gm", "VOUT", "VSS", "VIN", "VSS", "gm_eq")
            .resistor("rout", "VOUT", "VSS", "ro_eq")
            .build();

        Macro::new(
            "gain_stage",
            vec![
                MacroPort::new("VIN", MacroPortRole::Input),
                MacroPort::new("VOUT", MacroPortRole::Output),
                MacroPort::new("VSS", MacroPortRole::Ground),
            ],
            circuit,
            compact_model,
        )
    }

    #[test]
    fn groups_structure_compact_model_and_testbenches_in_one_macro() {
        let macro_ = gain_stage().with_ac_testbench(MacroAcTestbench::from_spice(
            "gain",
            "V1 VIN VSS 1\n",
            analysis(),
        ));

        assert_eq!(macro_.name(), "gain_stage");
        assert_eq!(macro_.ports().len(), 3);
        assert!(macro_.circuit().instance("xdp").is_some());
        assert!(macro_.compact_model().instance("gm").is_some());
        assert_eq!(macro_.exploration().testbenches().len(), 1);
        assert_eq!(
            macro_
                .exploration()
                .testbench("gain")
                .and_then(|testbench| testbench.source().as_spice()),
            Some("V1 VIN VSS 1\n")
        );
        assert_eq!(
            macro_.exploration().testbench("gain").unwrap().domain(),
            MacroAnalysisDomain::Electrical
        );
    }

    #[test]
    fn selects_layout_aware_analysis_explicitly() {
        let testbench = MacroAcTestbench::from_spice("layout", "V1 VIN VSS 1\n", analysis())
            .with_domain(MacroAnalysisDomain::LayoutAware);

        assert_eq!(testbench.domain(), MacroAnalysisDomain::LayoutAware);
    }

    #[test]
    fn supports_file_backed_testbenches_and_explicit_exploration() {
        let exploration = MacroExploration::new().with_ac_testbench(
            MacroAcTestbench::from_spice_file("gain", "testbenches/gain.spice", analysis()),
        );
        let macro_ = gain_stage().with_exploration(exploration);
        let testbench = macro_.exploration().testbench("gain").unwrap();

        assert_eq!(testbench.name(), "gain");
        assert_eq!(
            testbench.source().as_path(),
            Some(Path::new("testbenches/gain.spice"))
        );
        assert_eq!(testbench.analysis().transfer_function.input_node, "VIN");
    }
}
