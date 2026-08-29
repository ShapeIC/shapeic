//! Typed hierarchy paths, local runtime inputs, and parent-to-child conditions.

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt;

use shapeic_lut::DeviceLut;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::BlockRef;
use crate::primitive::build::PrimitiveBuildValue;

use super::specification::{accepted_candidate_symbols, evaluate_expression};
use super::{
    CompactMacroInstanceExplorationInput, Macro, MacroAnalysisDomain, MacroCatalog,
    MacroCompactSeedError, MacroDerivationReduction, MacroDerivationReductionError,
    MacroDerivationTarget, MacroDerivedValue, MacroDesignVariableCondition, MacroExecutionConfig,
    MacroExplorationInput, MacroExplorationInputRegistrationError,
    MacroExplorationInputValidationError, MacroExplorationResult, MacroHierarchyRetentionPolicy,
    MacroSpecificationBounds, MacroSpecificationEvaluationError, MacroValidationError,
    PrimitiveInstanceExplorationInput, validate_macro_catalog, validate_macro_exploration_input,
};

/// Stable path identifying one macro occurrence in a hierarchy.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct MacroHierarchyPath {
    components: Vec<String>,
}

impl MacroHierarchyPath {
    /// Creates the root path from the top-level macro name.
    pub fn root(name: impl Into<String>) -> Result<Self, MacroHierarchyPathError> {
        let name = name.into();
        validate_path_component(&name, 0)?;
        Ok(Self {
            components: vec![name],
        })
    }

    /// Appends one direct child instance to this path.
    pub fn child(&self, instance: impl Into<String>) -> Result<Self, MacroHierarchyPathError> {
        let instance = instance.into();
        validate_path_component(&instance, self.components.len())?;
        let mut components = self.components.clone();
        components.push(instance);
        Ok(Self { components })
    }

    /// Returns the root macro followed by local child-instance names.
    pub fn components(&self) -> &[String] {
        &self.components
    }

    /// Returns the containing macro path, or `None` for the root.
    pub fn parent(&self) -> Option<Self> {
        (self.components.len() > 1).then(|| Self {
            components: self.components[..self.components.len() - 1].to_vec(),
        })
    }
}

impl fmt::Display for MacroHierarchyPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.components.join("."))
    }
}

/// Invalid component in a typed hierarchy path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroHierarchyPathError {
    EmptyComponent { index: usize },
    AmbiguousComponent { index: usize, component: String },
}

impl fmt::Display for MacroHierarchyPathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyComponent { index } => {
                write!(formatter, "hierarchy path component {index} is empty")
            }
            Self::AmbiguousComponent { index, component } => write!(
                formatter,
                "hierarchy path component {index} ('{component}') contains '.'"
            ),
        }
    }
}

impl Error for MacroHierarchyPathError {}

fn validate_path_component(component: &str, index: usize) -> Result<(), MacroHierarchyPathError> {
    if component.trim().is_empty() {
        return Err(MacroHierarchyPathError::EmptyComponent { index });
    }
    if component.contains('.') {
        return Err(MacroHierarchyPathError::AmbiguousComponent {
            index,
            component: component.to_owned(),
        });
    }
    Ok(())
}

/// Runtime overrides owned by one concrete macro occurrence.
#[derive(Clone, Debug, Default)]
pub struct MacroHierarchyPathInput {
    primitive_instances: HashMap<String, PrimitiveInstanceExplorationInput>,
    design_variable_overrides: HashMap<String, PrimitiveBuildValue>,
    specification_overrides: HashMap<String, MacroSpecificationBounds>,
}

impl MacroHierarchyPathInput {
    /// Creates an empty path-local input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers runtime build data for one local primitive instance.
    pub fn register_primitive_instance(
        &mut self,
        instance: impl Into<String>,
        input: PrimitiveInstanceExplorationInput,
    ) -> Result<(), MacroHierarchyInputRegistrationError> {
        let instance = instance.into();
        if instance.trim().is_empty() {
            return Err(MacroHierarchyInputRegistrationError::EmptyInstance);
        }
        if self.primitive_instances.contains_key(&instance) {
            return Err(
                MacroHierarchyInputRegistrationError::DuplicatePrimitiveInstance { instance },
            );
        }
        self.primitive_instances.insert(instance, input);
        Ok(())
    }

    /// Overrides one public design variable only at this hierarchy path.
    pub fn register_design_variable_override(
        &mut self,
        variable: impl Into<String>,
        value: PrimitiveBuildValue,
    ) -> Result<(), MacroHierarchyInputRegistrationError> {
        let variable = variable.into();
        if variable.trim().is_empty() {
            return Err(MacroHierarchyInputRegistrationError::EmptyDesignVariable);
        }
        if self.design_variable_overrides.contains_key(&variable) {
            return Err(
                MacroHierarchyInputRegistrationError::DuplicateDesignVariableOverride { variable },
            );
        }
        self.design_variable_overrides.insert(variable, value);
        Ok(())
    }

    /// Overrides one specification only at this hierarchy path.
    pub fn register_specification_override(
        &mut self,
        specification: impl Into<String>,
        bounds: MacroSpecificationBounds,
    ) -> Result<(), MacroHierarchyInputRegistrationError> {
        let specification = specification.into();
        if specification.trim().is_empty() {
            return Err(MacroHierarchyInputRegistrationError::EmptySpecification);
        }
        if self.specification_overrides.contains_key(&specification) {
            return Err(
                MacroHierarchyInputRegistrationError::DuplicateSpecificationOverride {
                    specification,
                },
            );
        }
        self.specification_overrides.insert(specification, bounds);
        Ok(())
    }
}

/// Shared data plus independent runtime overrides for concrete hierarchy paths.
#[derive(Clone, Debug, Default)]
pub struct MacroHierarchyExplorationInput<'lut> {
    device_models: HashMap<String, &'lut DeviceLut>,
    paths: BTreeMap<MacroHierarchyPath, MacroHierarchyPathInput>,
    execution: MacroExecutionConfig,
    retention: MacroHierarchyRetentionPolicy,
}

impl<'lut> MacroHierarchyExplorationInput<'lut> {
    /// Creates an empty hierarchical exploration input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one read-only LUT shared by every hierarchy path.
    pub fn register_device_model(
        &mut self,
        device_type: impl Into<String>,
        model: &'lut DeviceLut,
    ) -> Result<(), MacroHierarchyInputRegistrationError> {
        let device_type = device_type.into();
        if device_type.trim().is_empty() {
            return Err(MacroHierarchyInputRegistrationError::EmptyDeviceType);
        }
        if self.device_models.contains_key(&device_type) {
            return Err(MacroHierarchyInputRegistrationError::DuplicateDeviceModel { device_type });
        }
        self.device_models.insert(device_type, model);
        Ok(())
    }

    /// Registers the runtime configuration of one concrete macro occurrence.
    pub fn register_path(
        &mut self,
        path: MacroHierarchyPath,
        input: MacroHierarchyPathInput,
    ) -> Result<(), MacroHierarchyInputRegistrationError> {
        if self.paths.contains_key(&path) {
            return Err(MacroHierarchyInputRegistrationError::DuplicatePath { path });
        }
        self.paths.insert(path, input);
        Ok(())
    }

    /// Returns the configuration registered for one concrete path.
    pub fn path(&self, path: &MacroHierarchyPath) -> Option<&MacroHierarchyPathInput> {
        self.paths.get(path)
    }

    /// Iterates through path-local inputs in deterministic path order.
    pub fn paths(&self) -> impl Iterator<Item = (&MacroHierarchyPath, &MacroHierarchyPathInput)> {
        self.paths.iter()
    }

    /// Replaces the execution policy shared by the hierarchy.
    pub fn set_execution_config(&mut self, execution: MacroExecutionConfig) {
        self.execution = execution;
    }

    /// Returns the execution policy shared by the hierarchy.
    pub const fn execution_config(&self) -> &MacroExecutionConfig {
        &self.execution
    }

    /// Replaces the policy controlling retention of complete preview results.
    pub fn set_retention_policy(&mut self, retention: MacroHierarchyRetentionPolicy) {
        self.retention = retention;
    }

    /// Returns the policy controlling retention of complete preview results.
    pub const fn retention_policy(&self) -> MacroHierarchyRetentionPolicy {
        self.retention
    }

    pub(super) fn local_input(
        &self,
        macro_: &Macro,
        macro_catalog: &MacroCatalog,
        path: &MacroHierarchyPath,
        derived: Option<&ResolvedChildConditions>,
        compact_children: Option<&BTreeMap<String, CompactMacroInstanceExplorationInput>>,
    ) -> Result<MacroExplorationInput<'lut>, Vec<MacroHierarchyLocalInputError>> {
        let mut input = MacroExplorationInput::new();
        input.set_execution_config(self.execution);
        let mut errors = Vec::new();
        for (device_type, model) in &self.device_models {
            register_local(
                input.register_device_model(device_type.clone(), *model),
                &mut errors,
            );
        }
        if let Some(path_input) = self.paths.get(path) {
            for (instance, primitive_input) in &path_input.primitive_instances {
                register_local(
                    input.register_primitive_instance(instance.clone(), primitive_input.clone()),
                    &mut errors,
                );
            }
            for (variable, value) in &path_input.design_variable_overrides {
                register_local(
                    input.register_design_variable_override(variable.clone(), value.clone()),
                    &mut errors,
                );
            }
            for (specification, bounds) in &path_input.specification_overrides {
                register_local(
                    input.register_specification_override(specification.clone(), *bounds),
                    &mut errors,
                );
            }
        }
        if let Some(derived) = derived {
            for (specification, bounds) in derived.specification_bounds() {
                register_local(
                    input.register_inherited_specification_bounds(specification.clone(), *bounds),
                    &mut errors,
                );
            }
            for (variable, condition) in derived.design_variable_conditions() {
                register_local(
                    input.register_design_variable_condition(variable.clone(), condition.clone()),
                    &mut errors,
                );
            }
        }
        for instance in macro_.circuit().instances() {
            let BlockRef::Macro(child_macro_name) = instance.block() else {
                continue;
            };
            let Some(child_macro) = macro_catalog.get(child_macro_name) else {
                continue;
            };
            let compact_input = if let Some(compact_children) = compact_children {
                compact_children
                    .get(instance.name())
                    .cloned()
                    .ok_or_else(|| {
                        vec![MacroHierarchyLocalInputError::MissingResolvedChild {
                            child_instance: instance.name().to_owned(),
                            child_macro: child_macro_name.clone(),
                        }]
                    })
            } else {
                CompactMacroInstanceExplorationInput::from_seed(
                    child_macro,
                    instance.name(),
                    Vec::new(),
                )
                .map_err(|seed_errors| {
                    seed_errors
                        .into_iter()
                        .map(|error| MacroHierarchyLocalInputError::CompactSeed {
                            child_instance: instance.name().to_owned(),
                            child_macro: child_macro_name.clone(),
                            error,
                        })
                        .collect::<Vec<_>>()
                })
            };
            match compact_input {
                Ok(seed) => register_local(
                    input.register_compact_macro_instance(instance.name(), seed),
                    &mut errors,
                ),
                Err(compact_errors) => errors.extend(compact_errors),
            }
        }
        if errors.is_empty() {
            Ok(input)
        } else {
            Err(errors)
        }
    }
}

fn register_local(
    result: Result<(), MacroExplorationInputRegistrationError>,
    errors: &mut Vec<MacroHierarchyLocalInputError>,
) {
    if let Err(error) = result {
        errors.push(MacroHierarchyLocalInputError::Registration(error));
    }
}

/// Errors while assembling hierarchical runtime configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroHierarchyInputRegistrationError {
    /// A device type is empty.
    EmptyDeviceType,
    /// A local primitive instance name is empty.
    EmptyInstance,
    /// A public design-variable name is empty.
    EmptyDesignVariable,
    /// A specification name is empty.
    EmptySpecification,
    /// A shared device model was registered more than once.
    DuplicateDeviceModel { device_type: String },
    /// A local primitive input was registered more than once.
    DuplicatePrimitiveInstance { instance: String },
    /// A design variable has more than one path-local override.
    DuplicateDesignVariableOverride { variable: String },
    /// A specification has more than one path-local override.
    DuplicateSpecificationOverride { specification: String },
    /// A concrete hierarchy path was registered more than once.
    DuplicatePath { path: MacroHierarchyPath },
}

impl fmt::Display for MacroHierarchyInputRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDeviceType => formatter.write_str("device type cannot be empty"),
            Self::EmptyInstance => formatter.write_str("primitive instance cannot be empty"),
            Self::EmptyDesignVariable => formatter.write_str("design variable cannot be empty"),
            Self::EmptySpecification => formatter.write_str("specification cannot be empty"),
            Self::DuplicateDeviceModel { device_type } => {
                write!(
                    formatter,
                    "device model '{device_type}' is already registered"
                )
            }
            Self::DuplicatePrimitiveInstance { instance } => write!(
                formatter,
                "primitive instance input '{instance}' is already registered"
            ),
            Self::DuplicateDesignVariableOverride { variable } => write!(
                formatter,
                "design variable '{variable}' already has a path-local override"
            ),
            Self::DuplicateSpecificationOverride { specification } => write!(
                formatter,
                "specification '{specification}' already has a path-local override"
            ),
            Self::DuplicatePath { path } => {
                write!(formatter, "hierarchy path '{path}' is already configured")
            }
        }
    }
}

impl Error for MacroHierarchyInputRegistrationError {}

/// Failure while translating one path-local configuration to a direct input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroHierarchyLocalInputError {
    /// A direct exploration-input registration failed.
    Registration(MacroExplorationInputRegistrationError),
    /// A direct child cannot be represented by its nominal compact seed.
    CompactSeed {
        child_instance: String,
        child_macro: String,
        error: MacroCompactSeedError,
    },
    /// A final parent refresh is missing one resolved child candidate set.
    MissingResolvedChild {
        child_instance: String,
        child_macro: String,
    },
}

impl fmt::Display for MacroHierarchyLocalInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registration(error) => error.fmt(formatter),
            Self::CompactSeed {
                child_instance,
                child_macro,
                error,
            } => write!(
                formatter,
                "child instance '{child_instance}' of macro '{child_macro}' has an invalid compact seed: {error}"
            ),
            Self::MissingResolvedChild {
                child_instance,
                child_macro,
            } => write!(
                formatter,
                "child instance '{child_instance}' of macro '{child_macro}' has no resolved candidate input"
            ),
        }
    }
}

impl Error for MacroHierarchyLocalInputError {}

/// Effective conditions inherited by one child occurrence.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolvedChildConditions {
    specification_bounds: BTreeMap<String, MacroSpecificationBounds>,
    design_variable_conditions: BTreeMap<String, MacroDesignVariableCondition>,
    audit: Vec<MacroDerivationAuditEntry>,
    pruned: Option<MacroDerivationPruneReason>,
}

impl ResolvedChildConditions {
    /// Returns derived specification bounds keyed by child-public name.
    pub fn specification_bounds(&self) -> &BTreeMap<String, MacroSpecificationBounds> {
        &self.specification_bounds
    }

    /// Returns derived pre-build conditions keyed by child-public variable.
    pub fn design_variable_conditions(&self) -> &BTreeMap<String, MacroDesignVariableCondition> {
        &self.design_variable_conditions
    }

    /// Returns the ordered trace of every applied derivation rule.
    pub fn audit(&self) -> &[MacroDerivationAuditEntry] {
        &self.audit
    }

    /// Returns why this child occurrence was pruned, when conditions conflict.
    pub const fn prune_reason(&self) -> Option<&MacroDerivationPruneReason> {
        self.pruned.as_ref()
    }

    /// Returns whether the derived condition intersection is empty.
    pub const fn is_pruned(&self) -> bool {
        self.pruned.is_some()
    }
}

/// Complete trace of one parent-to-child derivation rule.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroDerivationAuditEntry {
    rule_index: usize,
    child_instance: String,
    expression: String,
    source_rows: usize,
    reduction: MacroDerivationReduction,
    reduced_value: MacroDerivedValue,
    target: MacroDerivationTarget,
    effective_condition: MacroResolvedCondition,
}

impl MacroDerivationAuditEntry {
    /// Returns the rule's index in its parent definition.
    pub const fn rule_index(&self) -> usize {
        self.rule_index
    }
    /// Returns the direct child instance targeted by the rule.
    pub fn child_instance(&self) -> &str {
        &self.child_instance
    }
    /// Returns the expression evaluated for every accepted parent row.
    pub fn expression(&self) -> &str {
        &self.expression
    }
    /// Returns the number of accepted parent rows used by the reduction.
    pub const fn source_rows(&self) -> usize {
        self.source_rows
    }
    /// Returns the reduction applied to the row values.
    pub const fn reduction(&self) -> MacroDerivationReduction {
        self.reduction
    }
    /// Returns the value produced by the reduction.
    pub const fn reduced_value(&self) -> &MacroDerivedValue {
        &self.reduced_value
    }
    /// Returns the child-public target of the rule.
    pub const fn target(&self) -> &MacroDerivationTarget {
        &self.target
    }
    /// Returns the complete condition after intersecting this rule.
    pub const fn effective_condition(&self) -> &MacroResolvedCondition {
        &self.effective_condition
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MacroResolvedCondition {
    /// Effective bounds for one child specification, or an empty intersection.
    Specification {
        name: String,
        bounds: Option<MacroSpecificationBounds>,
    },
    /// Effective restriction for one design variable, or an empty intersection.
    DesignVariable {
        name: String,
        condition: Option<MacroDesignVariableCondition>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroDerivationPruneReason {
    /// The parent preview produced no rows from which to derive conditions.
    NoAcceptedParentCandidates,
    /// Derived limits for one specification do not overlap.
    ConflictingSpecification { specification: String },
    /// Derived conditions for one design variable do not overlap.
    ConflictingDesignVariable { variable: String },
}

/// Resolves all rules targeting one direct child from accepted parent rows.
pub fn resolve_child_derivations(
    parent: &Macro,
    parent_result: &MacroExplorationResult,
    child_instance: &str,
) -> Result<ResolvedChildConditions, MacroDerivationResolutionError> {
    if parent_result.macro_name() != parent.name() {
        return Err(MacroDerivationResolutionError::ResultMacroMismatch {
            expected: parent.name().to_owned(),
            actual: parent_result.macro_name().to_owned(),
        });
    }
    let instance = parent.circuit().instance(child_instance).ok_or_else(|| {
        MacroDerivationResolutionError::UnknownChildInstance {
            parent: parent.name().to_owned(),
            child_instance: child_instance.to_owned(),
        }
    })?;
    if !matches!(instance.block(), BlockRef::Macro(_)) {
        return Err(MacroDerivationResolutionError::ChildIsNotMacro {
            parent: parent.name().to_owned(),
            child_instance: child_instance.to_owned(),
        });
    }

    let rules = parent
        .exploration()
        .derivation_rules()
        .iter()
        .enumerate()
        .filter(|(_, rule)| rule.child_instance() == child_instance)
        .collect::<Vec<_>>();
    let mut resolved = ResolvedChildConditions::default();
    if rules.is_empty() {
        return Ok(resolved);
    }
    if parent_result.accepted().is_empty() {
        resolved.pruned = Some(MacroDerivationPruneReason::NoAcceptedParentCandidates);
        return Ok(resolved);
    }

    for (rule_index, rule) in rules {
        let values = parent_result
            .accepted()
            .iter()
            .map(|candidate| {
                let symbols =
                    accepted_candidate_symbols(parent, parent_result.candidate_sets(), candidate)
                        .map_err(|error| MacroDerivationResolutionError::ExpressionContext {
                        parent: parent.name().to_owned(),
                        child_instance: child_instance.to_owned(),
                        rule_index,
                        error,
                    })?;
                evaluate_expression(rule.expression(), &symbols).map_err(|reason| {
                    MacroDerivationResolutionError::Expression {
                        parent: parent.name().to_owned(),
                        child_instance: child_instance.to_owned(),
                        rule_index,
                        expression: rule.expression().to_owned(),
                        reason,
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reduced = rule.reduction().reduce(values).map_err(|error| {
            MacroDerivationResolutionError::Reduction {
                parent: parent.name().to_owned(),
                child_instance: child_instance.to_owned(),
                rule_index,
                error,
            }
        })?;
        let effective = apply_derived_condition(&mut resolved, rule.target(), &reduced);
        resolved.audit.push(MacroDerivationAuditEntry {
            rule_index,
            child_instance: child_instance.to_owned(),
            expression: rule.expression().to_owned(),
            source_rows: parent_result.accepted().len(),
            reduction: rule.reduction(),
            reduced_value: reduced,
            target: rule.target().clone(),
            effective_condition: effective,
        });
    }
    Ok(resolved)
}

fn apply_derived_condition(
    resolved: &mut ResolvedChildConditions,
    target: &MacroDerivationTarget,
    value: &MacroDerivedValue,
) -> MacroResolvedCondition {
    match (target, value) {
        (
            MacroDerivationTarget::SpecificationMinimum { specification },
            MacroDerivedValue::Scalar(value),
        ) => apply_specification(
            resolved,
            specification,
            MacroSpecificationBounds::at_least(*value),
        ),
        (
            MacroDerivationTarget::SpecificationMaximum { specification },
            MacroDerivedValue::Scalar(value),
        ) => apply_specification(
            resolved,
            specification,
            MacroSpecificationBounds::at_most(*value),
        ),
        (
            MacroDerivationTarget::SpecificationRange { specification },
            MacroDerivedValue::Range { minimum, maximum },
        ) => apply_specification(
            resolved,
            specification,
            MacroSpecificationBounds::between(*minimum, *maximum),
        ),
        (
            MacroDerivationTarget::DesignVariableRange { variable },
            MacroDerivedValue::Range { minimum, maximum },
        ) => apply_design_variable(
            resolved,
            variable,
            MacroDesignVariableCondition::range(Some(*minimum), Some(*maximum)),
        ),
        (
            MacroDerivationTarget::DesignVariableAllowedValues { variable },
            MacroDerivedValue::UniqueValues(values),
        ) => apply_design_variable(
            resolved,
            variable,
            MacroDesignVariableCondition::allowed_values(values.clone()),
        ),
        _ => unreachable!("derivation shapes are validated with their targets"),
    }
}

fn apply_specification(
    resolved: &mut ResolvedChildConditions,
    name: &str,
    incoming: MacroSpecificationBounds,
) -> MacroResolvedCondition {
    let effective = resolved
        .specification_bounds
        .get(name)
        .copied()
        .unwrap_or_else(MacroSpecificationBounds::unbounded)
        .intersection(incoming);
    match effective {
        Some(bounds) => {
            resolved
                .specification_bounds
                .insert(name.to_owned(), bounds);
        }
        None => {
            resolved.pruned.get_or_insert_with(|| {
                MacroDerivationPruneReason::ConflictingSpecification {
                    specification: name.to_owned(),
                }
            });
        }
    }
    MacroResolvedCondition::Specification {
        name: name.to_owned(),
        bounds: effective,
    }
}

fn apply_design_variable(
    resolved: &mut ResolvedChildConditions,
    name: &str,
    incoming: MacroDesignVariableCondition,
) -> MacroResolvedCondition {
    let effective = resolved
        .design_variable_conditions
        .get(name)
        .cloned()
        .map_or(Some(incoming.clone()), |current| {
            intersect_design_conditions(&current, &incoming)
        });
    match &effective {
        Some(condition) => {
            resolved
                .design_variable_conditions
                .insert(name.to_owned(), condition.clone());
        }
        None => {
            resolved.pruned.get_or_insert_with(|| {
                MacroDerivationPruneReason::ConflictingDesignVariable {
                    variable: name.to_owned(),
                }
            });
        }
    }
    MacroResolvedCondition::DesignVariable {
        name: name.to_owned(),
        condition: effective,
    }
}

fn intersect_design_conditions(
    left: &MacroDesignVariableCondition,
    right: &MacroDesignVariableCondition,
) -> Option<MacroDesignVariableCondition> {
    match (left, right) {
        (
            MacroDesignVariableCondition::Range {
                minimum: lmin,
                maximum: lmax,
            },
            MacroDesignVariableCondition::Range {
                minimum: rmin,
                maximum: rmax,
            },
        ) => {
            let minimum = match (lmin, rmin) {
                (Some(a), Some(b)) => Some(a.max(*b)),
                (a, b) => a.or(*b),
            };
            let maximum = match (lmax, rmax) {
                (Some(a), Some(b)) => Some(a.min(*b)),
                (a, b) => a.or(*b),
            };
            if minimum.zip(maximum).is_some_and(|(min, max)| min > max) {
                None
            } else {
                Some(MacroDesignVariableCondition::range(minimum, maximum))
            }
        }
        (
            MacroDesignVariableCondition::AllowedValues(left),
            MacroDesignVariableCondition::AllowedValues(right),
        ) => {
            let values = left
                .iter()
                .copied()
                .filter(|value| right.contains(value))
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| MacroDesignVariableCondition::allowed_values(values))
        }
        (
            MacroDesignVariableCondition::AllowedValues(values),
            MacroDesignVariableCondition::Range { minimum, maximum },
        )
        | (
            MacroDesignVariableCondition::Range { minimum, maximum },
            MacroDesignVariableCondition::AllowedValues(values),
        ) => {
            let values = values
                .iter()
                .copied()
                .filter(|value| {
                    minimum.is_none_or(|min| *value >= min)
                        && maximum.is_none_or(|max| *value <= max)
                })
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| MacroDesignVariableCondition::allowed_values(values))
        }
    }
}

#[derive(Debug)]
pub enum MacroDerivationResolutionError {
    ResultMacroMismatch {
        expected: String,
        actual: String,
    },
    UnknownChildInstance {
        parent: String,
        child_instance: String,
    },
    ChildIsNotMacro {
        parent: String,
        child_instance: String,
    },
    ExpressionContext {
        parent: String,
        child_instance: String,
        rule_index: usize,
        error: MacroSpecificationEvaluationError,
    },
    Expression {
        parent: String,
        child_instance: String,
        rule_index: usize,
        expression: String,
        reason: String,
    },
    Reduction {
        parent: String,
        child_instance: String,
        rule_index: usize,
        error: MacroDerivationReductionError,
    },
}

impl fmt::Display for MacroDerivationResolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResultMacroMismatch { expected, actual } => write!(
                formatter,
                "cannot derive conditions for macro '{expected}' from result for '{actual}'"
            ),
            Self::UnknownChildInstance {
                parent,
                child_instance,
            } => write!(
                formatter,
                "macro '{parent}' has no child instance '{child_instance}'"
            ),
            Self::ChildIsNotMacro {
                parent,
                child_instance,
            } => write!(
                formatter,
                "macro '{parent}' instance '{child_instance}' is not a submacro"
            ),
            Self::ExpressionContext {
                parent,
                child_instance,
                rule_index,
                error,
            } => write!(
                formatter,
                "macro '{parent}' derivation rule {rule_index} for child '{child_instance}' cannot build its expression context: {error}"
            ),
            Self::Expression {
                parent,
                child_instance,
                rule_index,
                expression,
                reason,
            } => write!(
                formatter,
                "macro '{parent}' derivation rule {rule_index} for child '{child_instance}' cannot evaluate '{expression}': {reason}"
            ),
            Self::Reduction {
                parent,
                child_instance,
                rule_index,
                error,
            } => write!(
                formatter,
                "macro '{parent}' derivation rule {rule_index} for child '{child_instance}' cannot reduce its values: {error}"
            ),
        }
    }
}
impl Error for MacroDerivationResolutionError {}

/// One path-qualified validation problem detected before hierarchical execution.
#[derive(Debug)]
pub enum MacroHierarchyValidationError {
    UnknownTopMacro {
        macro_name: String,
    },
    InvalidTopPath {
        macro_name: String,
        error: MacroHierarchyPathError,
    },
    UnsupportedAnalysisDomain {
        path: MacroHierarchyPath,
        macro_name: String,
        testbench: String,
        domain: MacroAnalysisDomain,
    },
    Catalog(MacroValidationError),
    UnknownConfiguredPath {
        path: MacroHierarchyPath,
    },
    LocalInput {
        path: MacroHierarchyPath,
        macro_name: String,
        error: MacroHierarchyLocalInputError,
    },
    InvalidInput {
        path: MacroHierarchyPath,
        macro_name: String,
        error: MacroExplorationInputValidationError,
    },
}

impl fmt::Display for MacroHierarchyValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTopMacro { macro_name } => {
                write!(formatter, "top macro '{macro_name}' is not registered")
            }
            Self::InvalidTopPath { macro_name, error } => write!(
                formatter,
                "top macro '{macro_name}' has an invalid hierarchy path: {error}"
            ),
            Self::UnsupportedAnalysisDomain {
                path,
                macro_name,
                testbench,
                domain,
            } => write!(
                formatter,
                "hierarchy path '{path}' (macro '{macro_name}') testbench '{testbench}' uses unsupported {domain:?} analysis"
            ),
            Self::Catalog(error) => write!(formatter, "invalid macro catalog: {error}"),
            Self::UnknownConfiguredPath { path } => {
                write!(
                    formatter,
                    "configured hierarchy path '{path}' is not reachable"
                )
            }
            Self::LocalInput {
                path,
                macro_name,
                error,
            } => write!(
                formatter,
                "hierarchy path '{path}' (macro '{macro_name}') has invalid local input: {error}"
            ),
            Self::InvalidInput {
                path,
                macro_name,
                error,
            } => write!(
                formatter,
                "hierarchy path '{path}' (macro '{macro_name}') has invalid exploration input: {error}"
            ),
        }
    }
}
impl Error for MacroHierarchyValidationError {}

/// Validates the complete reachable hierarchy and every configured occurrence.
pub fn validate_macro_hierarchy_input(
    top_macro: &str,
    macro_catalog: &MacroCatalog,
    primitive_catalog: &PrimitiveCatalog,
    input: &MacroHierarchyExplorationInput<'_>,
) -> Vec<MacroHierarchyValidationError> {
    let Some(top) = macro_catalog.get(top_macro) else {
        return vec![MacroHierarchyValidationError::UnknownTopMacro {
            macro_name: top_macro.to_owned(),
        }];
    };
    let mut errors = validate_macro_catalog(macro_catalog, primitive_catalog)
        .into_iter()
        .map(MacroHierarchyValidationError::Catalog)
        .collect::<Vec<_>>();
    let root = match MacroHierarchyPath::root(top_macro) {
        Ok(root) => root,
        Err(error) => {
            errors.push(MacroHierarchyValidationError::InvalidTopPath {
                macro_name: top_macro.to_owned(),
                error,
            });
            return errors;
        }
    };
    let mut reachable = BTreeMap::new();
    collect_paths(top, macro_catalog, root, &mut Vec::new(), &mut reachable);
    for configured in input.paths.keys() {
        if !reachable.contains_key(configured) {
            errors.push(MacroHierarchyValidationError::UnknownConfiguredPath {
                path: configured.clone(),
            });
        }
    }
    for (path, macro_) in reachable {
        errors.extend(
            macro_
                .exploration()
                .testbenches()
                .iter()
                .filter(|testbench| testbench.domain() != MacroAnalysisDomain::Electrical)
                .map(
                    |testbench| MacroHierarchyValidationError::UnsupportedAnalysisDomain {
                        path: path.clone(),
                        macro_name: macro_.name().to_owned(),
                        testbench: testbench.name().to_owned(),
                        domain: testbench.domain(),
                    },
                ),
        );
        match input.local_input(macro_, macro_catalog, &path, None, None) {
            Ok(local) => errors.extend(
                validate_macro_exploration_input(macro_, primitive_catalog, &local)
                    .into_iter()
                    .filter(|error| {
                        !matches!(
                            error,
                            MacroExplorationInputValidationError::MissingPhysicalLut { .. }
                        )
                    })
                    .map(|error| MacroHierarchyValidationError::InvalidInput {
                        path: path.clone(),
                        macro_name: macro_.name().to_owned(),
                        error,
                    }),
            ),
            Err(local_errors) => errors.extend(local_errors.into_iter().map(|error| {
                MacroHierarchyValidationError::LocalInput {
                    path: path.clone(),
                    macro_name: macro_.name().to_owned(),
                    error,
                }
            })),
        }
    }
    errors
}

fn collect_paths<'a>(
    macro_: &'a Macro,
    catalog: &'a MacroCatalog,
    path: MacroHierarchyPath,
    stack: &mut Vec<String>,
    paths: &mut BTreeMap<MacroHierarchyPath, &'a Macro>,
) {
    paths.insert(path.clone(), macro_);
    if stack.iter().any(|name| name == macro_.name()) {
        return;
    }
    stack.push(macro_.name().to_owned());
    for instance in macro_.circuit().instances() {
        if let BlockRef::Macro(child_name) = instance.block() {
            if let (Some(child), Ok(child_path)) =
                (catalog.get(child_name), path.child(instance.name()))
            {
                collect_paths(child, catalog, child_path, stack, paths);
            }
        }
    }
    stack.pop();
}

#[cfg(test)]
mod tests {
    use crate::catalog::primitive_catalog::PrimitiveCatalog;
    use crate::circuit::Circuit;
    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::exploration::filter::CandidateFilterReport;

    use super::super::{
        MacroCandidateSets, MacroDerivationRule, MacroExplorationInstanceKind,
        MacroInstanceCandidateSet,
    };
    use super::*;

    #[test]
    fn hierarchy_paths_are_typed_ordered_and_unambiguous() {
        let root = MacroHierarchyPath::root("ota").unwrap();
        let child = root.child("xstage").unwrap();
        assert_eq!(child.to_string(), "ota.xstage");
        assert_eq!(child.parent(), Some(root));
        assert!(MacroHierarchyPath::root("").is_err());
        assert!(child.child("x.bad").is_err());
    }

    #[test]
    fn condition_intersection_preserves_allowed_order_and_detects_conflicts() {
        let allowed = MacroDesignVariableCondition::allowed_values([3.0, 1.0, 2.0]);
        let range = MacroDesignVariableCondition::range(Some(1.5), Some(3.5));
        assert_eq!(
            intersect_design_conditions(&allowed, &range),
            Some(MacroDesignVariableCondition::allowed_values([3.0, 2.0]))
        );
        assert_eq!(
            intersect_design_conditions(
                &MacroDesignVariableCondition::range(Some(0.0), Some(1.0)),
                &MacroDesignVariableCondition::range(Some(2.0), Some(3.0)),
            ),
            None
        );
    }

    #[test]
    fn duplicate_registration_does_not_replace_the_original_path_input() {
        let path = MacroHierarchyPath::root("top").unwrap();
        let mut input = MacroHierarchyExplorationInput::new();
        input
            .register_path(path.clone(), MacroHierarchyPathInput::new())
            .unwrap();

        assert_eq!(
            input.register_path(path.clone(), MacroHierarchyPathInput::new()),
            Err(MacroHierarchyInputRegistrationError::DuplicatePath { path: path.clone() })
        );
        assert!(input.path(&path).is_some());
        assert_eq!(input.paths().count(), 1);
    }

    #[test]
    fn repeated_macro_instances_resolve_to_independent_paths() {
        let child = Macro::new(
            "child",
            Vec::new(),
            Circuit::builder().resistor("r", "a", "b", 1.0).build(),
            Circuit::builder().resistor("r", "a", "b", 1.0).build(),
        );
        let parent = Macro::new(
            "parent",
            Vec::new(),
            Circuit::builder()
                .macro_instance("xa", "child", Vec::<(&str, &str)>::new())
                .macro_instance("xb", "child", Vec::<(&str, &str)>::new())
                .build(),
            Circuit::builder().resistor("r", "a", "b", 1.0).build(),
        );
        let catalog = MacroCatalog::from_macros([child, parent]).unwrap();
        let root = MacroHierarchyPath::root("parent").unwrap();
        let mut paths = BTreeMap::new();
        collect_paths(
            catalog.get("parent").unwrap(),
            &catalog,
            root.clone(),
            &mut Vec::new(),
            &mut paths,
        );

        assert_eq!(
            paths.get(&root.child("xa").unwrap()).unwrap().name(),
            "child"
        );
        assert_eq!(
            paths.get(&root.child("xb").unwrap()).unwrap().name(),
            "child"
        );
        assert_eq!(paths.len(), 3);
    }

    #[test]
    fn validates_each_repeated_macro_occurrence_with_its_own_path_input() {
        let child = Macro::new(
            "child",
            Vec::new(),
            Circuit::builder().resistor("r", "a", "b", 1.0).build(),
            Circuit::builder().resistor("r", "a", "b", 1.0).build(),
        )
        .with_compact_seed(super::super::MacroCompactSeed::default());
        let parent = Macro::new(
            "parent",
            Vec::new(),
            Circuit::builder()
                .macro_instance("xa", "child", Vec::<(&str, &str)>::new())
                .macro_instance("xb", "child", Vec::<(&str, &str)>::new())
                .build(),
            Circuit::builder().resistor("r", "a", "b", 1.0).build(),
        );
        let catalog = MacroCatalog::from_macros([child, parent]).unwrap();
        let root = MacroHierarchyPath::root("parent").unwrap();
        let xa = root.child("xa").unwrap();
        let xb = root.child("xb").unwrap();
        let mut xa_input = MacroHierarchyPathInput::new();
        xa_input
            .register_design_variable_override("unknown", PrimitiveBuildValue::Scalar(1.0))
            .unwrap();
        let mut input = MacroHierarchyExplorationInput::new();
        input.register_path(xa.clone(), xa_input).unwrap();
        input
            .register_path(xb.clone(), MacroHierarchyPathInput::new())
            .unwrap();

        let errors =
            validate_macro_hierarchy_input("parent", &catalog, &PrimitiveCatalog::new(), &input);

        assert!(errors.iter().any(|error| matches!(
            error,
            MacroHierarchyValidationError::InvalidInput { path, .. } if path == &xa
        )));
        assert!(!errors.iter().any(|error| matches!(
            error,
            MacroHierarchyValidationError::InvalidInput { path, .. } if path == &xb
        )));
    }

    #[test]
    fn resolves_rules_from_accepted_rows_and_records_the_effective_intersection() {
        let parent = Macro::new(
            "parent",
            Vec::new(),
            Circuit::builder()
                .macro_instance("xchild", "child", Vec::<(&str, &str)>::new())
                .build(),
            Circuit::builder().build(),
        )
        .with_derivation_rule(MacroDerivationRule::new(
            "xchild",
            "score + margin",
            MacroDerivationReduction::Minimum,
            MacroDerivationTarget::specification_minimum("gain"),
        ))
        .with_derivation_rule(MacroDerivationRule::new(
            "xchild",
            "score + margin",
            MacroDerivationReduction::Maximum,
            MacroDerivationTarget::specification_maximum("gain"),
        ))
        .with_derivation_rule(MacroDerivationRule::new(
            "xchild",
            "bias",
            MacroDerivationReduction::UniqueValues,
            MacroDerivationTarget::design_variable_allowed_values("bias"),
        ));
        let candidate_sets = MacroCandidateSets {
            instances: vec![MacroInstanceCandidateSet {
                instance_path: "xsource".to_owned(),
                kind: MacroExplorationInstanceKind::Primitive,
                candidates: CandidateSet::new(
                    "source",
                    vec![
                        CandidatePoint::new(vec![
                            ("score".to_owned(), 2.0),
                            ("bias".to_owned(), 0.3),
                        ]),
                        CandidatePoint::new(vec![
                            ("score".to_owned(), 5.0),
                            ("bias".to_owned(), 0.7),
                        ]),
                    ],
                ),
                filter_report: CandidateFilterReport::default(),
                interface_ports: Vec::new(),
                compact_provenance: None,
            }],
        };
        let result = MacroExplorationResult::test_fixture(
            "parent",
            candidate_sets,
            vec![
                (vec![0], vec![("margin".to_owned(), 1.0)]),
                (vec![1], vec![("margin".to_owned(), 1.0)]),
            ],
        );

        let resolved = resolve_child_derivations(&parent, &result, "xchild").unwrap();

        assert_eq!(
            resolved.specification_bounds().get("gain"),
            Some(&MacroSpecificationBounds::between(3.0, 6.0))
        );
        assert_eq!(
            resolved.design_variable_conditions().get("bias"),
            Some(&MacroDesignVariableCondition::allowed_values([0.3, 0.7]))
        );
        assert_eq!(resolved.audit().len(), 3);
        assert!(
            resolved
                .audit()
                .iter()
                .all(|entry| entry.source_rows() == 2)
        );
        assert!(!resolved.is_pruned());
    }

    #[test]
    fn contradictory_derived_bounds_prune_the_child() {
        let mut resolved = ResolvedChildConditions::default();
        apply_specification(
            &mut resolved,
            "gain",
            MacroSpecificationBounds::at_least(10.0),
        );
        let effective = apply_specification(
            &mut resolved,
            "gain",
            MacroSpecificationBounds::at_most(5.0),
        );

        assert_eq!(
            effective,
            MacroResolvedCondition::Specification {
                name: "gain".to_owned(),
                bounds: None,
            }
        );
        assert_eq!(
            resolved.prune_reason(),
            Some(&MacroDerivationPruneReason::ConflictingSpecification {
                specification: "gain".to_owned()
            })
        );
    }
}
