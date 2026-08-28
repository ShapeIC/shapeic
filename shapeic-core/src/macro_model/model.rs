use std::path::{Path, PathBuf};

use crate::analysis::AcMetric;
use crate::circuit::Circuit;
use crate::exploration::filter::CandidateFilter;
use crate::primitive::build::{PrimitiveBuildInput, PrimitiveBuildInputKind};
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

    /// Adds one acceptance specification independent from its source analysis.
    pub fn with_specification(mut self, specification: MacroSpecification) -> Self {
        self.exploration.specifications.push(specification);
        self
    }

    /// Adds base build data and filters for one primitive implementation instance.
    pub fn with_primitive_default(mut self, default: MacroPrimitiveDefault) -> Self {
        self.exploration.primitive_defaults.push(default);
        self
    }

    /// Exposes one public design variable backed by primitive build inputs.
    pub fn with_design_variable(mut self, variable: MacroDesignVariable) -> Self {
        self.exploration.design_variables.push(variable);
        self
    }

    /// Sets the nominal compact point used before this macro is explored as a child.
    pub fn with_compact_seed(mut self, seed: MacroCompactSeed) -> Self {
        self.exploration.compact_seed = Some(seed);
        self
    }

    /// Adds one condition derived by this macro for a direct child instance.
    pub fn with_derivation_rule(mut self, rule: MacroDerivationRule) -> Self {
        self.exploration.derivation_rules.push(rule);
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
    specifications: Vec<MacroSpecification>,
    primitive_defaults: Vec<MacroPrimitiveDefault>,
    design_variables: Vec<MacroDesignVariable>,
    compact_seed: Option<MacroCompactSeed>,
    derivation_rules: Vec<MacroDerivationRule>,
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

    /// Adds one macro-level acceptance specification.
    pub fn with_specification(mut self, specification: MacroSpecification) -> Self {
        self.specifications.push(specification);
        self
    }

    /// Adds base build data and filters for one primitive instance.
    pub fn with_primitive_default(mut self, default: MacroPrimitiveDefault) -> Self {
        self.primitive_defaults.push(default);
        self
    }

    /// Adds one public design variable.
    pub fn with_design_variable(mut self, variable: MacroDesignVariable) -> Self {
        self.design_variables.push(variable);
        self
    }

    /// Sets the nominal compact point used by a parent preview.
    pub fn with_compact_seed(mut self, seed: MacroCompactSeed) -> Self {
        self.compact_seed = Some(seed);
        self
    }

    /// Adds one derivation rule targeting a direct child instance.
    pub fn with_derivation_rule(mut self, rule: MacroDerivationRule) -> Self {
        self.derivation_rules.push(rule);
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

    /// Returns macro specifications in declaration order.
    pub fn specifications(&self) -> &[MacroSpecification] {
        &self.specifications
    }

    /// Finds a macro specification by name.
    pub fn specification(&self, name: &str) -> Option<&MacroSpecification> {
        self.specifications
            .iter()
            .find(|specification| specification.name == name)
    }

    /// Returns primitive defaults in declaration order.
    pub fn primitive_defaults(&self) -> &[MacroPrimitiveDefault] {
        &self.primitive_defaults
    }

    /// Finds base build data for one local primitive instance path.
    pub fn primitive_default(&self, instance_path: &str) -> Option<&MacroPrimitiveDefault> {
        self.primitive_defaults
            .iter()
            .find(|default| default.instance_path == instance_path)
    }

    /// Returns public design variables in declaration order.
    pub fn design_variables(&self) -> &[MacroDesignVariable] {
        &self.design_variables
    }

    /// Finds one public design variable by name.
    pub fn design_variable(&self, name: &str) -> Option<&MacroDesignVariable> {
        self.design_variables
            .iter()
            .find(|variable| variable.name == name)
    }

    /// Returns the nominal compact point, when this macro can seed a parent preview.
    pub const fn compact_seed(&self) -> Option<&MacroCompactSeed> {
        self.compact_seed.as_ref()
    }

    /// Returns child derivation rules in declaration order.
    pub fn derivation_rules(&self) -> &[MacroDerivationRule] {
        &self.derivation_rules
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

/// Base candidate-build input and filters owned by a macro definition.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroPrimitiveDefault {
    instance_path: String,
    build_input: PrimitiveBuildInput,
    filters: Vec<CandidateFilter>,
}

impl MacroPrimitiveDefault {
    /// Creates base exploration data for one local primitive instance.
    pub fn new(
        instance_path: impl Into<String>,
        build_input: PrimitiveBuildInput,
        filters: Vec<CandidateFilter>,
    ) -> Self {
        Self {
            instance_path: instance_path.into(),
            build_input,
            filters,
        }
    }

    /// Returns the local primitive instance path.
    pub fn instance_path(&self) -> &str {
        &self.instance_path
    }

    /// Returns the base primitive build input.
    pub const fn build_input(&self) -> &PrimitiveBuildInput {
        &self.build_input
    }

    /// Returns filters applied before any runtime filters.
    pub fn filters(&self) -> &[CandidateFilter] {
        &self.filters
    }
}

/// One primitive input controlled by a public macro design variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroDesignVariableBinding {
    instance_path: String,
    input: String,
}

impl MacroDesignVariableBinding {
    /// Creates a binding to one local primitive build input.
    pub fn new(instance_path: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            instance_path: instance_path.into(),
            input: input.into(),
        }
    }

    /// Returns the local primitive instance path.
    pub fn instance_path(&self) -> &str {
        &self.instance_path
    }

    /// Returns the primitive build-input name.
    pub fn input(&self) -> &str {
        &self.input
    }
}

/// Public design variable that fans out to one or more primitive inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroDesignVariable {
    name: String,
    kind: PrimitiveBuildInputKind,
    bindings: Vec<MacroDesignVariableBinding>,
}

impl MacroDesignVariable {
    /// Creates a public variable with its required input type and bindings.
    pub fn new(
        name: impl Into<String>,
        kind: PrimitiveBuildInputKind,
        bindings: Vec<MacroDesignVariableBinding>,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            bindings,
        }
    }

    /// Returns the public variable name used by hierarchical rules.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the primitive input type required by every binding.
    pub const fn kind(&self) -> PrimitiveBuildInputKind {
        self.kind
    }

    /// Returns all primitive inputs controlled by this variable.
    pub fn bindings(&self) -> &[MacroDesignVariableBinding] {
        &self.bindings
    }
}

/// An inclusive pre-build restriction applied to a public design variable.
#[derive(Clone, Debug, PartialEq)]
pub enum MacroDesignVariableCondition {
    /// Keeps values inside the inclusive numerical interval.
    Range {
        minimum: Option<f64>,
        maximum: Option<f64>,
    },
    /// Keeps values exactly equal to one of the listed values.
    AllowedValues(Vec<f64>),
}

impl MacroDesignVariableCondition {
    /// Creates an inclusive range. An inverted range is a valid empty condition.
    pub const fn range(minimum: Option<f64>, maximum: Option<f64>) -> Self {
        Self::Range { minimum, maximum }
    }

    /// Creates an exact allowed-values condition. An empty list rejects every value.
    pub fn allowed_values(values: impl IntoIterator<Item = f64>) -> Self {
        Self::AllowedValues(values.into_iter().collect())
    }

    /// Returns whether one value satisfies this condition.
    pub fn accepts(&self, value: f64) -> bool {
        value.is_finite()
            && match self {
                Self::Range { minimum, maximum } => {
                    minimum.is_none_or(|minimum| value >= minimum)
                        && maximum.is_none_or(|maximum| value <= maximum)
                }
                Self::AllowedValues(values) => values.contains(&value),
            }
    }

    pub(super) fn is_finite(&self) -> bool {
        match self {
            Self::Range { minimum, maximum } => {
                minimum.is_none_or(f64::is_finite) && maximum.is_none_or(f64::is_finite)
            }
            Self::AllowedValues(values) => values.iter().all(|value| value.is_finite()),
        }
    }
}

/// One nominal parent-visible compact candidate for a macro used as a child.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MacroCompactSeed {
    compact_parameters: Vec<(String, f64)>,
    interface_values: Vec<(String, f64)>,
}

impl MacroCompactSeed {
    /// Creates a seed from unscoped compact parameter and public interface names.
    pub fn new<P, I, PN, IN>(compact_parameters: P, interface_values: I) -> Self
    where
        P: IntoIterator<Item = (PN, f64)>,
        I: IntoIterator<Item = (IN, f64)>,
        PN: Into<String>,
        IN: Into<String>,
    {
        Self {
            compact_parameters: compact_parameters
                .into_iter()
                .map(|(name, value)| (name.into(), value))
                .collect(),
            interface_values: interface_values
                .into_iter()
                .map(|(name, value)| (name.into(), value))
                .collect(),
        }
    }

    /// Returns unscoped compact-model parameter values.
    pub fn compact_parameters(&self) -> &[(String, f64)] {
        &self.compact_parameters
    }

    /// Returns unscoped public interface values.
    pub fn interface_values(&self) -> &[(String, f64)] {
        &self.interface_values
    }
}

/// Aggregation applied to one derivation expression over accepted parent rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroDerivationReduction {
    /// Selects the smallest expression value.
    Minimum,
    /// Selects the largest expression value.
    Maximum,
    /// Produces the smallest and largest expression values.
    Range,
    /// Preserves exact distinct values in first-occurrence order.
    UniqueValues,
}

impl MacroDerivationReduction {
    /// Reduces values evaluated from accepted parent candidates.
    pub fn reduce(
        self,
        values: impl IntoIterator<Item = f64>,
    ) -> Result<MacroDerivedValue, super::MacroDerivationReductionError> {
        super::derivation::reduce_derivation_values(self, values)
    }
}

/// Shape produced by a derivation reduction before it is applied to a target.
#[derive(Clone, Debug, PartialEq)]
pub enum MacroDerivedValue {
    /// One reduced scalar.
    Scalar(f64),
    /// Inclusive extrema of all source rows.
    Range { minimum: f64, maximum: f64 },
    /// Exact values in first-occurrence order.
    UniqueValues(Vec<f64>),
}

/// Public child condition populated by a derivation rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroDerivationTarget {
    /// Sets the child's effective specification minimum.
    SpecificationMinimum { specification: String },
    /// Sets the child's effective specification maximum.
    SpecificationMaximum { specification: String },
    /// Sets both bounds of a child specification.
    SpecificationRange { specification: String },
    /// Restricts a child design variable to an inclusive range.
    DesignVariableRange { variable: String },
    /// Restricts a child design variable to exact values.
    DesignVariableAllowedValues { variable: String },
}

impl MacroDerivationTarget {
    /// Targets the minimum of one child specification.
    pub fn specification_minimum(specification: impl Into<String>) -> Self {
        Self::SpecificationMinimum {
            specification: specification.into(),
        }
    }

    /// Targets the maximum of one child specification.
    pub fn specification_maximum(specification: impl Into<String>) -> Self {
        Self::SpecificationMaximum {
            specification: specification.into(),
        }
    }

    /// Targets the complete range of one child specification.
    pub fn specification_range(specification: impl Into<String>) -> Self {
        Self::SpecificationRange {
            specification: specification.into(),
        }
    }

    /// Targets an inclusive range on one child design variable.
    pub fn design_variable_range(variable: impl Into<String>) -> Self {
        Self::DesignVariableRange {
            variable: variable.into(),
        }
    }

    /// Targets exact allowed values on one child design variable.
    pub fn design_variable_allowed_values(variable: impl Into<String>) -> Self {
        Self::DesignVariableAllowedValues {
            variable: variable.into(),
        }
    }
}

/// One parent-owned rule that derives a public condition for a direct child.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroDerivationRule {
    child_instance: String,
    expression: String,
    reduction: MacroDerivationReduction,
    target: MacroDerivationTarget,
}

impl MacroDerivationRule {
    /// Creates a rule for one direct child instance.
    pub fn new(
        child_instance: impl Into<String>,
        expression: impl Into<String>,
        reduction: MacroDerivationReduction,
        target: MacroDerivationTarget,
    ) -> Self {
        Self {
            child_instance: child_instance.into(),
            expression: expression.into(),
            reduction,
            target,
        }
    }

    /// Returns the local child instance targeted by this rule.
    pub fn child_instance(&self) -> &str {
        &self.child_instance
    }

    /// Returns the expression evaluated on each accepted parent row.
    pub fn expression(&self) -> &str {
        &self.expression
    }

    /// Returns the aggregation applied to the expression values.
    pub const fn reduction(&self) -> MacroDerivationReduction {
        self.reduction
    }

    /// Returns the public child condition populated by the reduced value.
    pub const fn target(&self) -> &MacroDerivationTarget {
        &self.target
    }
}

/// Inclusive numerical limits applied independently from an analysis engine.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MacroSpecificationBounds {
    minimum: Option<f64>,
    maximum: Option<f64>,
}

impl MacroSpecificationBounds {
    /// Creates an unbounded specification that only exposes its value.
    pub const fn unbounded() -> Self {
        Self {
            minimum: None,
            maximum: None,
        }
    }

    /// Creates inclusive lower and upper limits.
    pub const fn new(minimum: Option<f64>, maximum: Option<f64>) -> Self {
        Self { minimum, maximum }
    }

    /// Creates an inclusive minimum.
    pub const fn at_least(minimum: f64) -> Self {
        Self::new(Some(minimum), None)
    }

    /// Creates an inclusive maximum.
    pub const fn at_most(maximum: f64) -> Self {
        Self::new(None, Some(maximum))
    }

    /// Creates an inclusive closed range.
    pub const fn between(minimum: f64, maximum: f64) -> Self {
        Self::new(Some(minimum), Some(maximum))
    }

    /// Returns the inclusive minimum, when configured.
    pub const fn minimum(self) -> Option<f64> {
        self.minimum
    }

    /// Returns the inclusive maximum, when configured.
    pub const fn maximum(self) -> Option<f64> {
        self.maximum
    }

    /// Returns whether a finite value lies inside the configured limits.
    pub fn accepts(self, value: f64) -> bool {
        value.is_finite()
            && self.minimum.is_none_or(|minimum| value >= minimum)
            && self.maximum.is_none_or(|maximum| value <= maximum)
    }

    pub(super) fn is_valid(self) -> bool {
        self.minimum.is_none_or(f64::is_finite)
            && self.maximum.is_none_or(f64::is_finite)
            && self
                .minimum
                .zip(self.maximum)
                .is_none_or(|(minimum, maximum)| minimum <= maximum)
    }

    pub(super) fn intersection(self, other: Self) -> Option<Self> {
        let minimum = match (self.minimum, other.minimum) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left, right) => left.or(right),
        };
        let maximum = match (self.maximum, other.maximum) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
        let intersection = Self::new(minimum, maximum);
        intersection.is_valid().then_some(intersection)
    }
}

/// Source evaluated to obtain one macro specification value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroSpecificationSource {
    /// Reads one exact column from a selected local candidate.
    CandidateColumn {
        /// Circuit instance path owning the candidate set.
        instance_path: String,
        /// Exact candidate column name.
        column: String,
    },
    /// Reads one metric produced by a macro-local AC testbench.
    AcMetric {
        /// Macro-local testbench name.
        testbench: String,
        /// Requested AC metric.
        metric: AcMetric,
    },
    /// Evaluates an arithmetic expression over candidate columns, AC metrics,
    /// and other named specifications.
    Expression(String),
}

impl MacroSpecificationSource {
    /// Selects one exact column from a circuit instance candidate.
    pub fn candidate_column(instance_path: impl Into<String>, column: impl Into<String>) -> Self {
        Self::CandidateColumn {
            instance_path: instance_path.into(),
            column: column.into(),
        }
    }

    /// Selects one metric from a macro-local AC testbench.
    pub fn ac_metric(testbench: impl Into<String>, metric: AcMetric) -> Self {
        Self::AcMetric {
            testbench: testbench.into(),
            metric,
        }
    }

    /// Creates an arithmetic expression source.
    pub fn expression(expression: impl Into<String>) -> Self {
        Self::Expression(expression.into())
    }
}

/// One reusable acceptance criterion owned by a macro.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroSpecification {
    name: String,
    source: MacroSpecificationSource,
    bounds: MacroSpecificationBounds,
}

impl MacroSpecification {
    /// Creates a named specification with an independent source and limits.
    pub fn new(
        name: impl Into<String>,
        source: MacroSpecificationSource,
        bounds: MacroSpecificationBounds,
    ) -> Self {
        Self {
            name: name.into(),
            source,
            bounds,
        }
    }

    /// Returns the macro-local specification name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the value source.
    pub const fn source(&self) -> &MacroSpecificationSource {
        &self.source
    }

    /// Returns the inclusive acceptance limits.
    pub const fn bounds(&self) -> MacroSpecificationBounds {
        self.bounds
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
