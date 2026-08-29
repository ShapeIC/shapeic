use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use shapeic_layout::PhysicalLookupTable;
use shapeic_lut::DeviceLut;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::BlockRef;
use crate::exploration::candidate::CandidateSet;
use crate::exploration::filter::CandidateFilter;
use crate::primitive::build::{PrimitiveBuildInput, PrimitiveBuildInputKind, PrimitiveBuildValue};

use super::prebuild::PrimitiveInputConditions;
use super::{
    Macro, MacroAnalysisDomain, MacroCandidateProjection, MacroDesignVariableCondition,
    MacroExecutionConfig, MacroExplorationResult, MacroSpecificationBounds,
};

/// Build data and local pre-exploration filters for one primitive instance.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrimitiveInstanceExplorationInput {
    pub(super) build_input: PrimitiveBuildInput,
    pub(super) filters: Vec<CandidateFilter>,
    pub(super) prebuild_conditions: PrimitiveInputConditions,
}

impl PrimitiveInstanceExplorationInput {
    /// Creates the exploration input for one primitive instance.
    pub fn new(build_input: PrimitiveBuildInput, filters: Vec<CandidateFilter>) -> Self {
        Self {
            build_input,
            filters,
            prebuild_conditions: HashMap::new(),
        }
    }

    /// Returns the values used to build the primitive candidates.
    pub const fn build_input(&self) -> &PrimitiveBuildInput {
        &self.build_input
    }

    /// Returns the filters applied locally after building the candidates.
    pub fn filters(&self) -> &[CandidateFilter] {
        &self.filters
    }
}

/// Precomputed candidates and local filters for one compact submacro instance.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactMacroInstanceExplorationInput {
    pub(super) candidates: CandidateSet,
    pub(super) filters: Vec<CandidateFilter>,
    pub(super) interface_ports: Vec<String>,
    pub(super) provenance: Option<CompactMacroCandidateProvenance>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CompactMacroCandidateProvenance {
    pub(super) source_result: Arc<MacroExplorationResult>,
    pub(super) accepted_indices: Vec<usize>,
}

impl CompactMacroInstanceExplorationInput {
    /// Creates the exploration input for one compact submacro instance.
    pub fn new(candidates: CandidateSet, filters: Vec<CandidateFilter>) -> Self {
        Self {
            candidates,
            filters,
            interface_ports: Vec::new(),
            provenance: None,
        }
    }

    /// Creates input from an explored child projection with recursive provenance.
    pub fn from_projection(
        projection: MacroCandidateProjection,
        filters: Vec<CandidateFilter>,
    ) -> Self {
        let (candidates, interface_ports, source_result, accepted_indices) =
            projection.into_parts();
        Self {
            candidates,
            filters,
            interface_ports,
            provenance: Some(CompactMacroCandidateProvenance {
                source_result,
                accepted_indices,
            }),
        }
    }

    /// Creates one nominal compact candidate for an unexplored child instance.
    pub fn from_seed(
        macro_: &Macro,
        instance_path: impl Into<String>,
        filters: Vec<CandidateFilter>,
    ) -> Result<Self, Vec<super::MacroCompactSeedError>> {
        super::seed::build_compact_seed_input(macro_, instance_path.into(), filters)
    }

    pub(super) fn empty_for_pruned_child(macro_: &Macro, instance_path: String) -> Self {
        Self {
            candidates: CandidateSet::new(instance_path, Vec::new()),
            filters: Vec::new(),
            interface_ports: macro_
                .exploration()
                .interface_bindings()
                .iter()
                .map(|binding| binding.port().to_owned())
                .collect(),
            provenance: None,
        }
    }

    /// Returns the candidate set projected by the explored child macro.
    pub const fn candidates(&self) -> &CandidateSet {
        &self.candidates
    }

    /// Returns the filters applied locally to the projected candidates.
    pub fn filters(&self) -> &[CandidateFilter] {
        &self.filters
    }

    /// Returns public ports whose columns carry compact interface values.
    pub fn interface_ports(&self) -> &[String] {
        &self.interface_ports
    }
}

/// Runtime data required to build the candidate sets of one macro.
///
/// Device LUTs are borrowed because they are shared, read-only inputs that may
/// be substantially larger than the per-instance exploration configuration.
#[derive(Clone, Debug, Default)]
pub struct MacroExplorationInput<'lut> {
    pub(super) device_models: HashMap<String, &'lut DeviceLut>,
    pub(super) physical_lut: Option<&'lut PhysicalLookupTable>,
    pub(super) primitive_instances: HashMap<String, PrimitiveInstanceExplorationInput>,
    pub(super) compact_macro_instances: HashMap<String, CompactMacroInstanceExplorationInput>,
    pub(super) design_variable_overrides: HashMap<String, PrimitiveBuildValue>,
    pub(super) design_variable_conditions: HashMap<String, Vec<MacroDesignVariableCondition>>,
    pub(super) specification_overrides: HashMap<String, MacroSpecificationBounds>,
    pub(super) inherited_specification_bounds: HashMap<String, MacroSpecificationBounds>,
    pub(super) execution: MacroExecutionConfig,
}

impl<'lut> MacroExplorationInput<'lut> {
    /// Creates an empty exploration input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one shared LUT under the device type used by primitive manifests.
    pub fn register_device_model(
        &mut self,
        device_type: impl Into<String>,
        model: &'lut DeviceLut,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let device_type = device_type.into();
        if device_type.trim().is_empty() {
            return Err(MacroExplorationInputRegistrationError::EmptyDeviceType);
        }
        if self.device_models.contains_key(&device_type) {
            return Err(
                MacroExplorationInputRegistrationError::DuplicateDeviceModel { device_type },
            );
        }
        self.device_models.insert(device_type, model);
        Ok(())
    }

    /// Registers the shared physical LUT used by layout-aware testbenches.
    pub fn register_physical_lut(
        &mut self,
        lut: &'lut PhysicalLookupTable,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        if self.physical_lut.is_some() {
            return Err(MacroExplorationInputRegistrationError::DuplicatePhysicalLut);
        }
        self.physical_lut = Some(lut);
        Ok(())
    }

    /// Registers build data and filters for one primitive instance path.
    pub fn register_primitive_instance(
        &mut self,
        instance_path: impl Into<String>,
        input: PrimitiveInstanceExplorationInput,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let instance_path = self.validate_new_instance_path(instance_path.into())?;
        self.primitive_instances.insert(instance_path, input);
        Ok(())
    }

    /// Overrides one public macro design variable for this execution.
    pub fn register_design_variable_override(
        &mut self,
        variable: impl Into<String>,
        value: PrimitiveBuildValue,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let variable = variable.into();
        if variable.trim().is_empty() {
            return Err(MacroExplorationInputRegistrationError::EmptyDesignVariable);
        }
        if self.design_variable_overrides.contains_key(&variable) {
            return Err(
                MacroExplorationInputRegistrationError::DuplicateDesignVariableOverride {
                    variable,
                },
            );
        }
        self.design_variable_overrides.insert(variable, value);
        Ok(())
    }

    /// Adds one inherited pre-build restriction to a public design variable.
    ///
    /// Multiple conditions are intersected in registration order.
    pub fn register_design_variable_condition(
        &mut self,
        variable: impl Into<String>,
        condition: MacroDesignVariableCondition,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let variable = variable.into();
        if variable.trim().is_empty() {
            return Err(MacroExplorationInputRegistrationError::EmptyDesignVariable);
        }
        self.design_variable_conditions
            .entry(variable)
            .or_default()
            .push(condition);
        Ok(())
    }

    /// Replaces the definition-owned bounds of one macro specification.
    pub fn register_specification_override(
        &mut self,
        specification: impl Into<String>,
        bounds: MacroSpecificationBounds,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let specification = specification.into();
        if specification.trim().is_empty() {
            return Err(MacroExplorationInputRegistrationError::EmptySpecification);
        }
        if self.specification_overrides.contains_key(&specification) {
            return Err(
                MacroExplorationInputRegistrationError::DuplicateSpecificationOverride {
                    specification,
                },
            );
        }
        self.specification_overrides.insert(specification, bounds);
        Ok(())
    }

    /// Adds bounds inherited from a parent exploration.
    ///
    /// Inherited bounds are intersected with the default or runtime override;
    /// they never widen a macro's effective specification.
    pub fn register_inherited_specification_bounds(
        &mut self,
        specification: impl Into<String>,
        bounds: MacroSpecificationBounds,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let specification = specification.into();
        if specification.trim().is_empty() {
            return Err(MacroExplorationInputRegistrationError::EmptySpecification);
        }
        if self
            .inherited_specification_bounds
            .contains_key(&specification)
        {
            return Err(
                MacroExplorationInputRegistrationError::DuplicateInheritedSpecificationBounds {
                    specification,
                },
            );
        }
        self.inherited_specification_bounds
            .insert(specification, bounds);
        Ok(())
    }

    /// Registers projected candidates and filters for one compact submacro path.
    pub fn register_compact_macro_instance(
        &mut self,
        instance_path: impl Into<String>,
        input: CompactMacroInstanceExplorationInput,
    ) -> Result<(), MacroExplorationInputRegistrationError> {
        let instance_path = self.validate_new_instance_path(instance_path.into())?;
        self.compact_macro_instances.insert(instance_path, input);
        Ok(())
    }

    /// Returns the LUT registered for one primitive device type.
    pub fn device_model(&self, device_type: &str) -> Option<&'lut DeviceLut> {
        self.device_models.get(device_type).copied()
    }

    /// Returns the physical LUT shared by layout-aware testbenches.
    pub const fn physical_lut(&self) -> Option<&'lut PhysicalLookupTable> {
        self.physical_lut
    }

    /// Replaces the runtime execution policy for this exploration.
    pub fn set_execution_config(&mut self, execution: MacroExecutionConfig) {
        self.execution = execution;
    }

    /// Returns the runtime execution policy for this exploration.
    pub const fn execution_config(&self) -> &MacroExecutionConfig {
        &self.execution
    }

    /// Returns the input registered for one primitive instance path.
    pub fn primitive_instance(
        &self,
        instance_path: &str,
    ) -> Option<&PrimitiveInstanceExplorationInput> {
        self.primitive_instances.get(instance_path)
    }

    /// Returns the compact input registered for one submacro instance path.
    pub fn compact_macro_instance(
        &self,
        instance_path: &str,
    ) -> Option<&CompactMacroInstanceExplorationInput> {
        self.compact_macro_instances.get(instance_path)
    }

    /// Returns one runtime design-variable override.
    pub fn design_variable_override(&self, variable: &str) -> Option<&PrimitiveBuildValue> {
        self.design_variable_overrides.get(variable)
    }

    /// Returns pre-build restrictions registered for one public variable.
    pub fn design_variable_conditions(&self, variable: &str) -> &[MacroDesignVariableCondition] {
        self.design_variable_conditions
            .get(variable)
            .map_or(&[], Vec::as_slice)
    }

    /// Returns one runtime specification-bounds override.
    pub fn specification_override(&self, specification: &str) -> Option<MacroSpecificationBounds> {
        self.specification_overrides.get(specification).copied()
    }

    pub(super) fn resolve_primitive_defaults(&mut self, macro_: &Macro) {
        let runtime = self.primitive_instances.clone();
        let mut resolved = macro_
            .exploration()
            .primitive_defaults()
            .iter()
            .map(|default| {
                (
                    default.instance_path().to_owned(),
                    PrimitiveInstanceExplorationInput::new(
                        default.build_input().clone(),
                        default.filters().to_vec(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();

        for (instance_path, runtime_input) in &runtime {
            merge_primitive_instance_input(
                resolved.entry(instance_path.clone()).or_default(),
                runtime_input,
            );
        }
        for (variable_name, value) in &self.design_variable_overrides {
            if let Some(variable) = macro_.exploration().design_variable(variable_name) {
                for binding in variable.bindings() {
                    if let Some(input) = resolved.get_mut(binding.instance_path()) {
                        input
                            .build_input
                            .values
                            .insert(binding.input().to_owned(), value.clone());
                    }
                }
            }
        }
        for (instance_path, runtime_input) in &runtime {
            let resolved_input = resolved
                .get_mut(instance_path)
                .expect("runtime primitive input was inserted during default resolution");
            for (name, value) in &runtime_input.build_input.values {
                resolved_input
                    .build_input
                    .values
                    .insert(name.clone(), value.clone());
            }
            if runtime_input.build_input.lut_config.is_some() {
                resolved_input.build_input.lut_config =
                    runtime_input.build_input.lut_config.clone();
            }
        }
        for (variable_name, conditions) in &self.design_variable_conditions {
            if let Some(variable) = macro_.exploration().design_variable(variable_name) {
                for binding in variable.bindings() {
                    if let Some(input) = resolved.get_mut(binding.instance_path()) {
                        input
                            .prebuild_conditions
                            .entry(binding.input().to_owned())
                            .or_default()
                            .extend(conditions.iter().cloned());
                    }
                }
            }
        }
        self.primitive_instances = resolved;
    }

    pub(super) fn effective_specification_bounds(
        &self,
        macro_: &Macro,
    ) -> Result<Vec<MacroSpecificationBounds>, MacroExplorationInputValidationError> {
        macro_
            .exploration()
            .specifications()
            .iter()
            .map(|specification| {
                let base = self
                    .specification_overrides
                    .get(specification.name())
                    .copied()
                    .unwrap_or_else(|| specification.bounds());
                if !base.is_valid() {
                    return Err(
                        MacroExplorationInputValidationError::InvalidSpecificationBounds {
                            macro_name: macro_.name().to_owned(),
                            specification: specification.name().to_owned(),
                        },
                    );
                }
                let effective = if let Some(inherited) = self
                    .inherited_specification_bounds
                    .get(specification.name())
                    .copied()
                {
                    if !inherited.is_valid() {
                        return Err(
                            MacroExplorationInputValidationError::InvalidSpecificationBounds {
                                macro_name: macro_.name().to_owned(),
                                specification: specification.name().to_owned(),
                            },
                        );
                    }
                    base.intersection(inherited).ok_or_else(|| {
                        MacroExplorationInputValidationError::ConflictingSpecificationBounds {
                            macro_name: macro_.name().to_owned(),
                            specification: specification.name().to_owned(),
                        }
                    })?
                } else {
                    base
                };
                if !effective.is_valid() {
                    return Err(
                        MacroExplorationInputValidationError::InvalidSpecificationBounds {
                            macro_name: macro_.name().to_owned(),
                            specification: specification.name().to_owned(),
                        },
                    );
                }
                Ok(effective)
            })
            .collect()
    }

    fn validate_new_instance_path(
        &self,
        instance_path: String,
    ) -> Result<String, MacroExplorationInputRegistrationError> {
        if instance_path.trim().is_empty() {
            return Err(MacroExplorationInputRegistrationError::EmptyInstancePath);
        }
        if self.primitive_instances.contains_key(&instance_path)
            || self.compact_macro_instances.contains_key(&instance_path)
        {
            return Err(
                MacroExplorationInputRegistrationError::DuplicateInstanceInput { instance_path },
            );
        }
        Ok(instance_path)
    }
}

fn merge_primitive_instance_input(
    base: &mut PrimitiveInstanceExplorationInput,
    overlay: &PrimitiveInstanceExplorationInput,
) {
    base.build_input.values.extend(
        overlay
            .build_input
            .values
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    );
    if overlay.build_input.lut_config.is_some() {
        base.build_input.lut_config = overlay.build_input.lut_config.clone();
    }
    base.filters.extend(overlay.filters.iter().cloned());
    for (input, conditions) in &overlay.prebuild_conditions {
        base.prebuild_conditions
            .entry(input.clone())
            .or_default()
            .extend(conditions.iter().cloned());
    }
}

/// Errors produced while registering runtime macro exploration inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroExplorationInputRegistrationError {
    EmptyDeviceType,
    EmptyInstancePath,
    DuplicateDeviceModel { device_type: String },
    DuplicatePhysicalLut,
    DuplicateInstanceInput { instance_path: String },
    EmptyDesignVariable,
    EmptySpecification,
    DuplicateDesignVariableOverride { variable: String },
    DuplicateSpecificationOverride { specification: String },
    DuplicateInheritedSpecificationBounds { specification: String },
}

impl fmt::Display for MacroExplorationInputRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDeviceType => formatter.write_str("device type cannot be empty"),
            Self::EmptyInstancePath => formatter.write_str("instance path cannot be empty"),
            Self::DuplicateDeviceModel { device_type } => {
                write!(
                    formatter,
                    "device model '{device_type}' is already registered"
                )
            }
            Self::DuplicatePhysicalLut => {
                formatter.write_str("a physical LUT is already registered")
            }
            Self::DuplicateInstanceInput { instance_path } => write!(
                formatter,
                "exploration input for instance '{instance_path}' is already registered"
            ),
            Self::EmptyDesignVariable => formatter.write_str("design variable cannot be empty"),
            Self::EmptySpecification => formatter.write_str("specification cannot be empty"),
            Self::DuplicateDesignVariableOverride { variable } => write!(
                formatter,
                "design variable '{variable}' already has a runtime override"
            ),
            Self::DuplicateSpecificationOverride { specification } => write!(
                formatter,
                "specification '{specification}' already has a runtime override"
            ),
            Self::DuplicateInheritedSpecificationBounds { specification } => write!(
                formatter,
                "specification '{specification}' already has inherited bounds"
            ),
        }
    }
}

impl Error for MacroExplorationInputRegistrationError {}

/// Kind of circuit instance expected by a macro exploration input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroExplorationInstanceKind {
    Primitive,
    CompactMacro,
    LinearElement,
}

impl fmt::Display for MacroExplorationInstanceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primitive => formatter.write_str("primitive"),
            Self::CompactMacro => formatter.write_str("compact macro"),
            Self::LinearElement => formatter.write_str("linear element"),
        }
    }
}

/// One invalid or missing item in the runtime exploration input of a macro.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroExplorationInputValidationError {
    MissingPhysicalLut {
        macro_name: String,
        testbench: String,
    },
    UnknownInstance {
        macro_name: String,
        instance_path: String,
    },
    InstanceKindMismatch {
        macro_name: String,
        instance_path: String,
        expected: MacroExplorationInstanceKind,
        actual: MacroExplorationInstanceKind,
    },
    MissingPrimitiveInput {
        macro_name: String,
        instance_path: String,
        primitive: String,
    },
    MissingCompactMacroInput {
        macro_name: String,
        instance_path: String,
        referenced_macro: String,
    },
    UnknownPrimitive {
        macro_name: String,
        instance_path: String,
        primitive: String,
    },
    MissingPrimitiveDeviceType {
        macro_name: String,
        instance_path: String,
        primitive: String,
    },
    MissingDeviceModel {
        macro_name: String,
        instance_path: String,
        primitive: String,
        device_type: String,
    },
    UnknownDesignVariable {
        macro_name: String,
        variable: String,
    },
    DesignVariableKindMismatch {
        macro_name: String,
        variable: String,
        expected: PrimitiveBuildInputKind,
        actual: PrimitiveBuildInputKind,
    },
    NonFiniteDesignVariable {
        macro_name: String,
        variable: String,
    },
    InvalidDesignVariableCondition {
        macro_name: String,
        variable: String,
    },
    UnknownSpecificationOverride {
        macro_name: String,
        specification: String,
    },
    InvalidSpecificationBounds {
        macro_name: String,
        specification: String,
    },
    ConflictingSpecificationBounds {
        macro_name: String,
        specification: String,
    },
    UnknownPrimitiveInput {
        macro_name: String,
        instance_path: String,
        primitive: String,
        input: String,
    },
    MissingRequiredPrimitiveInput {
        macro_name: String,
        instance_path: String,
        primitive: String,
        input: String,
    },
    PrimitiveInputKindMismatch {
        macro_name: String,
        instance_path: String,
        primitive: String,
        input: String,
        expected: PrimitiveBuildInputKind,
        actual: PrimitiveBuildInputKind,
    },
    NonFinitePrimitiveInput {
        macro_name: String,
        instance_path: String,
        primitive: String,
        input: String,
    },
}

impl fmt::Display for MacroExplorationInputValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPhysicalLut {
                macro_name,
                testbench,
            } => write!(
                formatter,
                "macro '{macro_name}' layout-aware testbench '{testbench}' requires a physical LUT"
            ),
            Self::UnknownInstance {
                macro_name,
                instance_path,
            } => write!(
                formatter,
                "macro '{macro_name}' has exploration input for unknown instance '{instance_path}'"
            ),
            Self::InstanceKindMismatch {
                macro_name,
                instance_path,
                expected,
                actual,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' is a {actual}, but its exploration input expects a {expected}"
            ),
            Self::MissingPrimitiveInput {
                macro_name,
                instance_path,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' primitive instance '{instance_path}' ('{primitive}') has no build input"
            ),
            Self::MissingCompactMacroInput {
                macro_name,
                instance_path,
                referenced_macro,
            } => write!(
                formatter,
                "macro '{macro_name}' submacro instance '{instance_path}' ('{referenced_macro}') has no compact candidate input"
            ),
            Self::UnknownPrimitive {
                macro_name,
                instance_path,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' references unknown primitive '{primitive}'"
            ),
            Self::MissingPrimitiveDeviceType {
                macro_name,
                instance_path,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' has no device type"
            ),
            Self::MissingDeviceModel {
                macro_name,
                instance_path,
                primitive,
                device_type,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' requires missing device model '{device_type}'"
            ),
            Self::UnknownDesignVariable {
                macro_name,
                variable,
            } => write!(
                formatter,
                "macro '{macro_name}' has an override for unknown design variable '{variable}'"
            ),
            Self::DesignVariableKindMismatch {
                macro_name,
                variable,
                expected,
                actual,
            } => write!(
                formatter,
                "macro '{macro_name}' design variable '{variable}' expects {expected}, found {actual}"
            ),
            Self::NonFiniteDesignVariable {
                macro_name,
                variable,
            } => write!(
                formatter,
                "macro '{macro_name}' design variable '{variable}' contains a non-finite value"
            ),
            Self::InvalidDesignVariableCondition {
                macro_name,
                variable,
            } => write!(
                formatter,
                "macro '{macro_name}' design variable '{variable}' has a non-finite pre-build condition"
            ),
            Self::UnknownSpecificationOverride {
                macro_name,
                specification,
            } => write!(
                formatter,
                "macro '{macro_name}' has bounds for unknown specification '{specification}'"
            ),
            Self::InvalidSpecificationBounds {
                macro_name,
                specification,
            } => write!(
                formatter,
                "macro '{macro_name}' specification '{specification}' has invalid effective bounds"
            ),
            Self::ConflictingSpecificationBounds {
                macro_name,
                specification,
            } => write!(
                formatter,
                "macro '{macro_name}' specification '{specification}' has contradictory inherited bounds"
            ),
            Self::UnknownPrimitiveInput {
                macro_name,
                instance_path,
                primitive,
                input,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' has no build input named '{input}'"
            ),
            Self::MissingRequiredPrimitiveInput {
                macro_name,
                instance_path,
                primitive,
                input,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' is missing required build input '{input}'"
            ),
            Self::PrimitiveInputKindMismatch {
                macro_name,
                instance_path,
                primitive,
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' input '{input}' expects {expected}, found {actual}"
            ),
            Self::NonFinitePrimitiveInput {
                macro_name,
                instance_path,
                primitive,
                input,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance_path}' primitive '{primitive}' input '{input}' contains a non-finite value"
            ),
        }
    }
}

impl Error for MacroExplorationInputValidationError {}

/// Validates runtime exploration inputs against the implementation circuit.
///
/// Every independent problem is returned so callers can report a complete,
/// instance-qualified configuration error before candidate construction starts.
pub fn validate_macro_exploration_input(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    input: &MacroExplorationInput<'_>,
) -> Vec<MacroExplorationInputValidationError> {
    let mut errors = Vec::new();
    let mut effective_input = input.clone();
    effective_input.resolve_primitive_defaults(macro_);
    let input = &effective_input;

    for (variable_name, value) in &input.design_variable_overrides {
        let Some(variable) = macro_.exploration().design_variable(variable_name) else {
            errors.push(
                MacroExplorationInputValidationError::UnknownDesignVariable {
                    macro_name: macro_.name().to_owned(),
                    variable: variable_name.clone(),
                },
            );
            continue;
        };
        if !variable.kind().accepts(value.kind()) {
            errors.push(
                MacroExplorationInputValidationError::DesignVariableKindMismatch {
                    macro_name: macro_.name().to_owned(),
                    variable: variable_name.clone(),
                    expected: variable.kind(),
                    actual: value.kind(),
                },
            );
        }
        if !value.is_finite() {
            errors.push(
                MacroExplorationInputValidationError::NonFiniteDesignVariable {
                    macro_name: macro_.name().to_owned(),
                    variable: variable_name.clone(),
                },
            );
        }
    }

    for (variable_name, conditions) in &input.design_variable_conditions {
        if macro_
            .exploration()
            .design_variable(variable_name)
            .is_none()
        {
            let error = MacroExplorationInputValidationError::UnknownDesignVariable {
                macro_name: macro_.name().to_owned(),
                variable: variable_name.clone(),
            };
            if !errors.contains(&error) {
                errors.push(error);
            }
            continue;
        }
        if conditions.iter().any(|condition| !condition.is_finite()) {
            errors.push(
                MacroExplorationInputValidationError::InvalidDesignVariableCondition {
                    macro_name: macro_.name().to_owned(),
                    variable: variable_name.clone(),
                },
            );
        }
    }

    for (specification, bounds) in input
        .specification_overrides
        .iter()
        .chain(&input.inherited_specification_bounds)
    {
        if macro_.exploration().specification(specification).is_none() {
            errors.push(
                MacroExplorationInputValidationError::UnknownSpecificationOverride {
                    macro_name: macro_.name().to_owned(),
                    specification: specification.clone(),
                },
            );
        } else if !bounds.is_valid() {
            errors.push(
                MacroExplorationInputValidationError::InvalidSpecificationBounds {
                    macro_name: macro_.name().to_owned(),
                    specification: specification.clone(),
                },
            );
        }
    }
    if let Err(error) = input.effective_specification_bounds(macro_) {
        if !errors.contains(&error) {
            errors.push(error);
        }
    }

    if input.physical_lut.is_none() {
        errors.extend(
            macro_
                .exploration()
                .testbenches()
                .iter()
                .filter(|testbench| testbench.domain() == MacroAnalysisDomain::LayoutAware)
                .map(
                    |testbench| MacroExplorationInputValidationError::MissingPhysicalLut {
                        macro_name: macro_.name().to_owned(),
                        testbench: testbench.name().to_owned(),
                    },
                ),
        );
    }

    for instance_path in input.primitive_instances.keys() {
        validate_registered_instance_kind(
            macro_,
            instance_path,
            MacroExplorationInstanceKind::Primitive,
            &mut errors,
        );
    }
    for instance_path in input.compact_macro_instances.keys() {
        validate_registered_instance_kind(
            macro_,
            instance_path,
            MacroExplorationInstanceKind::CompactMacro,
            &mut errors,
        );
    }

    for instance in macro_.circuit().instances() {
        match instance.block() {
            BlockRef::Primitive(primitive) => {
                if !input.primitive_instances.contains_key(instance.name()) {
                    errors.push(
                        MacroExplorationInputValidationError::MissingPrimitiveInput {
                            macro_name: macro_.name().to_owned(),
                            instance_path: instance.name().to_owned(),
                            primitive: primitive.clone(),
                        },
                    );
                }

                let Some(manifest) = primitive_catalog.get(primitive) else {
                    errors.push(MacroExplorationInputValidationError::UnknownPrimitive {
                        macro_name: macro_.name().to_owned(),
                        instance_path: instance.name().to_owned(),
                        primitive: primitive.clone(),
                    });
                    continue;
                };
                let Some(device_type) = manifest
                    .transistor_type
                    .as_deref()
                    .filter(|device_type| !device_type.trim().is_empty())
                else {
                    errors.push(
                        MacroExplorationInputValidationError::MissingPrimitiveDeviceType {
                            macro_name: macro_.name().to_owned(),
                            instance_path: instance.name().to_owned(),
                            primitive: primitive.clone(),
                        },
                    );
                    continue;
                };
                if !input.device_models.contains_key(device_type) {
                    errors.push(MacroExplorationInputValidationError::MissingDeviceModel {
                        macro_name: macro_.name().to_owned(),
                        instance_path: instance.name().to_owned(),
                        primitive: primitive.clone(),
                        device_type: device_type.to_owned(),
                    });
                }
                if let (Some(instance_input), Some(build_spec)) = (
                    input.primitive_instances.get(instance.name()),
                    &manifest.build,
                ) {
                    validate_primitive_build_input(
                        macro_.name(),
                        instance.name(),
                        primitive,
                        build_spec,
                        &instance_input.build_input,
                        &mut errors,
                    );
                }
            }
            BlockRef::Macro(referenced_macro) => {
                if !input.compact_macro_instances.contains_key(instance.name()) {
                    errors.push(
                        MacroExplorationInputValidationError::MissingCompactMacroInput {
                            macro_name: macro_.name().to_owned(),
                            instance_path: instance.name().to_owned(),
                            referenced_macro: referenced_macro.clone(),
                        },
                    );
                }
            }
            BlockRef::Element(_) => {}
        }
    }

    errors
}

fn validate_primitive_build_input(
    macro_name: &str,
    instance_path: &str,
    primitive: &str,
    build_spec: &crate::primitive::build::PrimitiveBuildSpec,
    input: &PrimitiveBuildInput,
    errors: &mut Vec<MacroExplorationInputValidationError>,
) {
    for (name, value) in &input.values {
        let Some(input_spec) = build_spec.inputs.iter().find(|input| input.name == *name) else {
            errors.push(
                MacroExplorationInputValidationError::UnknownPrimitiveInput {
                    macro_name: macro_name.to_owned(),
                    instance_path: instance_path.to_owned(),
                    primitive: primitive.to_owned(),
                    input: name.clone(),
                },
            );
            continue;
        };
        if !input_spec.kind.accepts(value.kind()) {
            errors.push(
                MacroExplorationInputValidationError::PrimitiveInputKindMismatch {
                    macro_name: macro_name.to_owned(),
                    instance_path: instance_path.to_owned(),
                    primitive: primitive.to_owned(),
                    input: name.clone(),
                    expected: input_spec.kind,
                    actual: value.kind(),
                },
            );
        }
        if !value.is_finite() {
            errors.push(
                MacroExplorationInputValidationError::NonFinitePrimitiveInput {
                    macro_name: macro_name.to_owned(),
                    instance_path: instance_path.to_owned(),
                    primitive: primitive.to_owned(),
                    input: name.clone(),
                },
            );
        }
    }
    for input_spec in &build_spec.inputs {
        if input_spec.required && !input.values.contains_key(&input_spec.name) {
            errors.push(
                MacroExplorationInputValidationError::MissingRequiredPrimitiveInput {
                    macro_name: macro_name.to_owned(),
                    instance_path: instance_path.to_owned(),
                    primitive: primitive.to_owned(),
                    input: input_spec.name.clone(),
                },
            );
        }
    }
}

fn validate_registered_instance_kind(
    macro_: &Macro,
    instance_path: &str,
    expected: MacroExplorationInstanceKind,
    errors: &mut Vec<MacroExplorationInputValidationError>,
) {
    let Some(instance) = macro_.circuit().instance(instance_path) else {
        errors.push(MacroExplorationInputValidationError::UnknownInstance {
            macro_name: macro_.name().to_owned(),
            instance_path: instance_path.to_owned(),
        });
        return;
    };
    let actual = match instance.block() {
        BlockRef::Primitive(_) => MacroExplorationInstanceKind::Primitive,
        BlockRef::Macro(_) => MacroExplorationInstanceKind::CompactMacro,
        BlockRef::Element(_) => MacroExplorationInstanceKind::LinearElement,
    };
    if actual != expected {
        errors.push(MacroExplorationInputValidationError::InstanceKindMismatch {
            macro_name: macro_.name().to_owned(),
            instance_path: instance_path.to_owned(),
            expected,
            actual,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use shapeic_lut::LookupTable;

    use crate::analysis::{
        AcMetricSet, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets,
    };
    use crate::circuit::Circuit;
    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::exploration::filter::CandidateFilter;
    use crate::macro_model::{
        MacroDesignVariable, MacroDesignVariableBinding, MacroPrimitiveDefault, MacroSpecification,
        MacroSpecificationSource,
    };
    use crate::primitive::build::{
        PrimitiveBuildInputKind, PrimitiveBuildInputSpec, PrimitiveBuildSpec, PrimitiveBuildValue,
        SweepMode,
    };
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};
    use crate::testbench::{AcAnalysis, TransferFunction};

    use super::*;

    fn primitive(name: &str, device_type: Option<&str>) -> PrimitiveManifest {
        PrimitiveManifest {
            name: name.to_owned(),
            version: "1.0".to_owned(),
            description: None,
            subckt_name: name.to_owned(),
            pins: vec![Pin {
                name: "OUT".to_owned(),
                role: PinRole::Output,
            }],
            files: PrimitiveFiles {
                netlist: "netlist.spice".to_owned(),
                build: None,
                symbol: None,
            },
            small_signal: None,
            physical_model: None,
            transistor_type: device_type.map(str::to_owned),
            layout_params: None,
            lut_config: None,
            build: None,
        }
    }

    fn macro_with_primitive_and_submacro() -> Macro {
        Macro::new(
            "ota",
            Vec::new(),
            Circuit::builder()
                .primitive("xdp", "diffpair", [("OUT", "VOUT")])
                .macro_instance("xload", "active_load", [("OUT", "VOUT")])
                .resistor("rout", "VOUT", "0", 1.0e6)
                .build(),
            Circuit::builder()
                .resistor("rout", "VOUT", "0", 1.0e6)
                .build(),
        )
    }

    fn ac_analysis() -> AcAnalysis {
        AcAnalysis::new(
            TransferFunction::new("VIN", "VOUT"),
            AdaptiveAcConfig {
                min_frequency_hz: 1.0,
                max_frequency_hz: 1.0e6,
                coarse_points_per_decade: 4,
                crossing_relative_tolerance: 1.0e-3,
                max_refinement_steps: 16,
                retain_samples: false,
            },
            AdaptiveAcPolicy {
                mode: AnalysisMode::Prune,
                targets: AnalysisTargets::NONE,
                metrics: AcMetricSet::ALL,
            },
        )
    }

    fn one_candidate(name: &str) -> CandidateSet {
        CandidateSet::new(
            name,
            vec![CandidatePoint::new(vec![("gm".to_owned(), 1.0e-3)])],
        )
    }

    fn fixture() -> LookupTable {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../shapeic-lut/tests/fixtures/sstadex_combined.npz");
        LookupTable::open(path).expect("LUT fixture should load")
    }

    fn macro_with_defaults_and_design_variable() -> Macro {
        let default_input = |current| {
            PrimitiveBuildInput::new(HashMap::from([
                ("current".to_owned(), PrimitiveBuildValue::Scalar(current)),
                (
                    "vout".to_owned(),
                    PrimitiveBuildValue::Vector(vec![0.5, 1.0]),
                ),
            ]))
        };
        Macro::new(
            "defaults",
            Vec::new(),
            Circuit::builder()
                .primitive("x1", "device", [("OUT", "N1")])
                .primitive("x2", "device", [("OUT", "N2")])
                .build(),
            Circuit::builder().resistor("r", "N1", "0", 1.0).build(),
        )
        .with_primitive_default(MacroPrimitiveDefault::new(
            "x1",
            default_input(1.0),
            vec![CandidateFilter::at_least("x1.width", 1.0).unwrap()],
        ))
        .with_primitive_default(MacroPrimitiveDefault::new(
            "x2",
            default_input(1.0),
            Vec::new(),
        ))
        .with_design_variable(MacroDesignVariable::new(
            "current",
            PrimitiveBuildInputKind::Scalar,
            vec![
                MacroDesignVariableBinding::new("x1", "current"),
                MacroDesignVariableBinding::new("x2", "current"),
            ],
        ))
    }

    #[test]
    fn registers_borrowed_models_and_typed_instance_inputs() {
        let table = fixture();
        let nmos = table.model("fixture_nmos").unwrap();
        let mut input = MacroExplorationInput::new();

        input.register_device_model("nmos", nmos).unwrap();
        input
            .register_primitive_instance(
                "xdp",
                PrimitiveInstanceExplorationInput::new(PrimitiveBuildInput::default(), Vec::new()),
            )
            .unwrap();
        input
            .register_compact_macro_instance(
                "xload",
                CompactMacroInstanceExplorationInput::new(one_candidate("active_load"), Vec::new()),
            )
            .unwrap();

        assert_eq!(
            input.device_model("nmos").map(DeviceLut::name),
            Some("fixture_nmos")
        );
        assert!(input.primitive_instance("xdp").is_some());
        assert!(input.compact_macro_instance("xload").is_some());
        assert_eq!(
            input.register_primitive_instance(
                "xload",
                PrimitiveInstanceExplorationInput::new(PrimitiveBuildInput::default(), Vec::new()),
            ),
            Err(
                MacroExplorationInputRegistrationError::DuplicateInstanceInput {
                    instance_path: "xload".to_owned(),
                }
            )
        );
    }

    #[test]
    fn registers_only_one_shared_physical_lut() {
        let first = crate::macro_model::physical::tests::physical_lut_for("pair", &["P", "N"]);
        let second = crate::macro_model::physical::tests::physical_lut_for("pair", &["P", "N"]);
        let mut input = MacroExplorationInput::new();

        input.register_physical_lut(&first).unwrap();
        assert!(std::ptr::eq(input.physical_lut().unwrap(), &first));
        assert_eq!(
            input.register_physical_lut(&second),
            Err(MacroExplorationInputRegistrationError::DuplicatePhysicalLut)
        );
        assert!(std::ptr::eq(input.physical_lut().unwrap(), &first));
    }

    #[test]
    fn stores_an_explicit_execution_policy() {
        use crate::macro_model::{ElectricalAnalysisExecution, MacroExecutionConfig};

        let mut input = MacroExplorationInput::new();
        assert_eq!(
            input.execution_config().electrical_analysis(),
            ElectricalAnalysisExecution::Sequential
        );

        input.set_execution_config(
            MacroExecutionConfig::sequential()
                .with_parallel_electrical_analysis(4)
                .unwrap()
                .with_electrical_batch_size(11)
                .unwrap(),
        );
        assert_eq!(
            input.execution_config().electrical_analysis(),
            ElectricalAnalysisExecution::Parallel {
                workers: 4,
                batch_size: 11,
            }
        );
    }

    #[test]
    fn requires_a_physical_lut_for_layout_aware_testbenches() {
        let macro_ = macro_with_primitive_and_submacro().with_ac_testbench(
            super::super::MacroAcTestbench::from_spice("layout", "", ac_analysis())
                .with_domain(MacroAnalysisDomain::LayoutAware),
        );
        let errors = validate_macro_exploration_input(
            &macro_,
            &PrimitiveCatalog::new(),
            &MacroExplorationInput::new(),
        );

        assert!(
            errors.contains(&MacroExplorationInputValidationError::MissingPhysicalLut {
                macro_name: "ota".to_owned(),
                testbench: "layout".to_owned(),
            })
        );
    }

    #[test]
    fn accepts_a_complete_input_for_the_implementation_circuit() {
        let table = fixture();
        let nmos = table.model("fixture_nmos").unwrap();
        let macro_ = macro_with_primitive_and_submacro();
        let mut primitives = PrimitiveCatalog::new();
        primitives.register(primitive("diffpair", Some("nmos")));
        let mut input = MacroExplorationInput::new();
        input.register_device_model("nmos", nmos).unwrap();
        input
            .register_primitive_instance(
                "xdp",
                PrimitiveInstanceExplorationInput::new(PrimitiveBuildInput::default(), Vec::new()),
            )
            .unwrap();
        input
            .register_compact_macro_instance(
                "xload",
                CompactMacroInstanceExplorationInput::new(one_candidate("active_load"), Vec::new()),
            )
            .unwrap();

        assert!(validate_macro_exploration_input(&macro_, &primitives, &input).is_empty());
    }

    #[test]
    fn reports_all_missing_unknown_and_mistyped_instance_inputs() {
        let macro_ = macro_with_primitive_and_submacro();
        let mut primitives = PrimitiveCatalog::new();
        primitives.register(primitive("diffpair", Some("nmos")));
        let mut input = MacroExplorationInput::new();
        input
            .register_compact_macro_instance(
                "xdp",
                CompactMacroInstanceExplorationInput::new(one_candidate("wrong"), Vec::new()),
            )
            .unwrap();
        input
            .register_primitive_instance(
                "ghost",
                PrimitiveInstanceExplorationInput::new(PrimitiveBuildInput::default(), Vec::new()),
            )
            .unwrap();

        let errors = validate_macro_exploration_input(&macro_, &primitives, &input);

        assert!(errors.contains(
            &MacroExplorationInputValidationError::InstanceKindMismatch {
                macro_name: "ota".to_owned(),
                instance_path: "xdp".to_owned(),
                expected: MacroExplorationInstanceKind::CompactMacro,
                actual: MacroExplorationInstanceKind::Primitive,
            }
        ));
        assert!(
            errors.contains(&MacroExplorationInputValidationError::UnknownInstance {
                macro_name: "ota".to_owned(),
                instance_path: "ghost".to_owned(),
            })
        );
        assert!(errors.contains(
            &MacroExplorationInputValidationError::MissingPrimitiveInput {
                macro_name: "ota".to_owned(),
                instance_path: "xdp".to_owned(),
                primitive: "diffpair".to_owned(),
            }
        ));
        assert!(
            errors.contains(&MacroExplorationInputValidationError::MissingDeviceModel {
                macro_name: "ota".to_owned(),
                instance_path: "xdp".to_owned(),
                primitive: "diffpair".to_owned(),
                device_type: "nmos".to_owned(),
            })
        );
        assert!(errors.contains(
            &MacroExplorationInputValidationError::MissingCompactMacroInput {
                macro_name: "ota".to_owned(),
                instance_path: "xload".to_owned(),
                referenced_macro: "active_load".to_owned(),
            }
        ));
    }

    #[test]
    fn resolves_defaults_design_variables_and_direct_overrides_by_precedence() {
        let macro_ = macro_with_defaults_and_design_variable();
        let mut input = MacroExplorationInput::new();
        input
            .register_design_variable_override("current", PrimitiveBuildValue::Scalar(2.0))
            .unwrap();
        input
            .register_design_variable_condition(
                "current",
                MacroDesignVariableCondition::range(Some(1.0), Some(4.0)),
            )
            .unwrap();
        input
            .register_primitive_instance(
                "x1",
                PrimitiveInstanceExplorationInput::new(
                    PrimitiveBuildInput::new(HashMap::from([(
                        "current".to_owned(),
                        PrimitiveBuildValue::Scalar(3.0),
                    )])),
                    vec![CandidateFilter::at_most("x1.width", 10.0).unwrap()],
                ),
            )
            .unwrap();

        input.resolve_primitive_defaults(&macro_);

        assert_eq!(
            input.primitive_instance("x1").unwrap().build_input().values["current"],
            PrimitiveBuildValue::Scalar(3.0)
        );
        assert_eq!(
            input.primitive_instance("x2").unwrap().build_input().values["current"],
            PrimitiveBuildValue::Scalar(2.0)
        );
        assert_eq!(
            input.primitive_instance("x1").unwrap().build_input().values["vout"],
            PrimitiveBuildValue::Vector(vec![0.5, 1.0])
        );
        assert_eq!(input.primitive_instance("x1").unwrap().filters().len(), 2);
        assert_eq!(
            input.primitive_instance("x1").unwrap().prebuild_conditions["current"].len(),
            1
        );
        assert_eq!(
            input.primitive_instance("x2").unwrap().prebuild_conditions["current"].len(),
            1
        );
    }

    #[test]
    fn resolves_specification_override_before_inherited_intersection() {
        let macro_ =
            macro_with_defaults_and_design_variable().with_specification(MacroSpecification::new(
                "gain",
                MacroSpecificationSource::expression("1"),
                MacroSpecificationBounds::between(0.0, 100.0),
            ));
        let mut input = MacroExplorationInput::new();
        input
            .register_specification_override("gain", MacroSpecificationBounds::between(10.0, 90.0))
            .unwrap();
        input
            .register_inherited_specification_bounds(
                "gain",
                MacroSpecificationBounds::between(20.0, 80.0),
            )
            .unwrap();

        assert_eq!(
            input.effective_specification_bounds(&macro_).unwrap(),
            [MacroSpecificationBounds::between(20.0, 80.0)]
        );
    }

    #[test]
    fn validates_runtime_variable_and_primitive_input_types() {
        let macro_ = macro_with_defaults_and_design_variable();
        let mut primitive = primitive("device", Some("nmos"));
        primitive.build = Some(PrimitiveBuildSpec {
            inputs: vec![
                PrimitiveBuildInputSpec {
                    name: "current".to_owned(),
                    kind: PrimitiveBuildInputKind::Scalar,
                    required: true,
                    source: None,
                },
                PrimitiveBuildInputSpec {
                    name: "vout".to_owned(),
                    kind: PrimitiveBuildInputKind::Vector,
                    required: true,
                    source: None,
                },
            ],
            sweep_mode: SweepMode::Cartesian,
            derived: Vec::new(),
            lut: Vec::new(),
            columns: Vec::new(),
        });
        let mut primitives = PrimitiveCatalog::new();
        primitives.register(primitive);
        let mut input = MacroExplorationInput::new();
        input
            .register_design_variable_override("current", PrimitiveBuildValue::Vector(vec![1.0]))
            .unwrap();
        input
            .register_design_variable_condition(
                "current",
                MacroDesignVariableCondition::range(Some(f64::NAN), None),
            )
            .unwrap();

        let errors = validate_macro_exploration_input(&macro_, &primitives, &input);

        assert!(errors.iter().any(|error| matches!(
            error,
            MacroExplorationInputValidationError::DesignVariableKindMismatch {
                variable,
                ..
            } if variable == "current"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroExplorationInputValidationError::InvalidDesignVariableCondition {
                variable,
                ..
            } if variable == "current"
        )));
    }
}
