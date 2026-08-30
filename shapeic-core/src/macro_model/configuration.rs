//! Validation of definition-owned primitive defaults and public design variables.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::BlockRef;
use crate::primitive::build::PrimitiveBuildInputKind;

use super::{
    Macro, MacroCatalog, MacroCompactSeedError, MacroDerivationReduction, MacroDerivationTarget,
    MacroHierarchyMode,
};

/// One invalid primitive default or public design-variable declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroExplorationDefinitionError {
    EmptyPrimitiveDefaultInstance {
        index: usize,
    },
    DuplicatePrimitiveDefault {
        instance_path: String,
    },
    UnknownPrimitiveDefaultInstance {
        instance_path: String,
    },
    PrimitiveDefaultTargetsNonPrimitive {
        instance_path: String,
    },
    UnknownPrimitiveDefaultInput {
        instance_path: String,
        input: String,
    },
    PrimitiveDefaultInputKindMismatch {
        instance_path: String,
        input: String,
        expected: PrimitiveBuildInputKind,
        actual: PrimitiveBuildInputKind,
    },
    NonFinitePrimitiveDefaultInput {
        instance_path: String,
        input: String,
    },
    EmptyDesignVariable {
        index: usize,
    },
    DuplicateDesignVariable {
        variable: String,
    },
    EmptyDesignVariableBindings {
        variable: String,
    },
    EmptyDesignVariableBinding {
        variable: String,
        binding_index: usize,
        field: &'static str,
    },
    DuplicateDesignVariableBinding {
        instance_path: String,
        input: String,
    },
    UnknownDesignVariableInstance {
        variable: String,
        instance_path: String,
    },
    DesignVariableTargetsNonPrimitive {
        variable: String,
        instance_path: String,
    },
    UnknownDesignVariableInput {
        variable: String,
        instance_path: String,
        input: String,
    },
    DesignVariableInputKindMismatch {
        variable: String,
        instance_path: String,
        input: String,
        expected: PrimitiveBuildInputKind,
        actual: PrimitiveBuildInputKind,
    },
    EmptyPublicInputAlias {
        alias_index: usize,
        field: &'static str,
    },
    DuplicatePublicInputAlias {
        child_instance: String,
        child_variable: String,
    },
    UnknownPublicInputAliasVariable {
        variable: String,
    },
    UnknownPublicInputAliasInstance {
        child_instance: String,
    },
    PublicInputAliasTargetsNonMacro {
        child_instance: String,
    },
    UnknownPublicInputAliasChildMacro {
        child_macro: String,
    },
    UnknownPublicInputAliasChildVariable {
        child_macro: String,
        variable: String,
    },
    PublicInputAliasKindMismatch {
        variable: String,
        child_macro: String,
        child_variable: String,
    },
    PublicInputAliasMissingParentNetBinding {
        variable: String,
        child_instance: String,
        interface_port: String,
        net: String,
    },
    UnknownPublicInputAliasPort {
        child_macro: String,
        port: String,
    },
    InvalidCompactSeed {
        error: MacroCompactSeedError,
    },
    EmptyDerivationChild {
        rule_index: usize,
    },
    EmptyDerivationExpression {
        rule_index: usize,
    },
    InvalidDerivationExpression {
        rule_index: usize,
        reason: String,
    },
    UnknownDerivationChild {
        rule_index: usize,
        child_instance: String,
    },
    DerivationTargetsNonMacro {
        rule_index: usize,
        child_instance: String,
    },
    EmptyDerivationTarget {
        rule_index: usize,
    },
    IncompatibleDerivationTarget {
        rule_index: usize,
    },
    UnknownDerivationChildMacro {
        rule_index: usize,
        child_macro: String,
    },
    UnknownDerivedSpecification {
        rule_index: usize,
        child_macro: String,
        specification: String,
    },
    UnknownDerivedDesignVariable {
        rule_index: usize,
        child_macro: String,
        variable: String,
    },
}

impl fmt::Display for MacroExplorationDefinitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrimitiveDefaultInstance { index } => {
                write!(
                    formatter,
                    "primitive default at index {index} has an empty instance path"
                )
            }
            Self::DuplicatePrimitiveDefault { instance_path } => write!(
                formatter,
                "primitive instance '{instance_path}' has more than one default"
            ),
            Self::UnknownPrimitiveDefaultInstance { instance_path } => write!(
                formatter,
                "primitive default references unknown instance '{instance_path}'"
            ),
            Self::PrimitiveDefaultTargetsNonPrimitive { instance_path } => write!(
                formatter,
                "default target '{instance_path}' is not a primitive instance"
            ),
            Self::UnknownPrimitiveDefaultInput {
                instance_path,
                input,
            } => write!(
                formatter,
                "primitive default for '{instance_path}' references unknown input '{input}'"
            ),
            Self::PrimitiveDefaultInputKindMismatch {
                instance_path,
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "primitive default for '{instance_path}.{input}' expects {expected}, found {actual}"
            ),
            Self::NonFinitePrimitiveDefaultInput {
                instance_path,
                input,
            } => write!(
                formatter,
                "primitive default for '{instance_path}.{input}' contains a non-finite value"
            ),
            Self::EmptyDesignVariable { index } => {
                write!(
                    formatter,
                    "design variable at index {index} has an empty name"
                )
            }
            Self::DuplicateDesignVariable { variable } => {
                write!(
                    formatter,
                    "design variable '{variable}' is declared more than once"
                )
            }
            Self::EmptyDesignVariableBindings { variable } => {
                write!(formatter, "design variable '{variable}' has no bindings")
            }
            Self::EmptyDesignVariableBinding {
                variable,
                binding_index,
                field,
            } => write!(
                formatter,
                "design variable '{variable}' binding {binding_index} has an empty {field}"
            ),
            Self::DuplicateDesignVariableBinding {
                instance_path,
                input,
            } => write!(
                formatter,
                "primitive input '{instance_path}.{input}' is bound by more than one design variable"
            ),
            Self::UnknownDesignVariableInstance {
                variable,
                instance_path,
            } => write!(
                formatter,
                "design variable '{variable}' references unknown instance '{instance_path}'"
            ),
            Self::DesignVariableTargetsNonPrimitive {
                variable,
                instance_path,
            } => write!(
                formatter,
                "design variable '{variable}' target '{instance_path}' is not a primitive instance"
            ),
            Self::UnknownDesignVariableInput {
                variable,
                instance_path,
                input,
            } => write!(
                formatter,
                "design variable '{variable}' references unknown input '{instance_path}.{input}'"
            ),
            Self::DesignVariableInputKindMismatch {
                variable,
                instance_path,
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "design variable '{variable}' declares {actual}, but '{instance_path}.{input}' expects {expected}"
            ),
            Self::EmptyPublicInputAlias { alias_index, field } => write!(
                formatter,
                "public input alias {alias_index} has an empty {field}"
            ),
            Self::DuplicatePublicInputAlias {
                child_instance,
                child_variable,
            } => write!(
                formatter,
                "child input alias '{child_instance}.{child_variable}' is declared more than once"
            ),
            Self::UnknownPublicInputAliasVariable { variable } => write!(
                formatter,
                "public input alias references unknown parent variable '{variable}'"
            ),
            Self::UnknownPublicInputAliasInstance { child_instance } => write!(
                formatter,
                "public input alias references unknown child instance '{child_instance}'"
            ),
            Self::PublicInputAliasTargetsNonMacro { child_instance } => write!(
                formatter,
                "public input alias target '{child_instance}' is not a macro instance"
            ),
            Self::UnknownPublicInputAliasChildMacro { child_macro } => write!(
                formatter,
                "public input alias references unregistered child macro '{child_macro}'"
            ),
            Self::UnknownPublicInputAliasChildVariable {
                child_macro,
                variable,
            } => write!(
                formatter,
                "public input alias references unknown variable '{variable}' in child macro '{child_macro}'"
            ),
            Self::PublicInputAliasKindMismatch {
                variable,
                child_macro,
                child_variable,
            } => write!(
                formatter,
                "parent variable '{variable}' is incompatible with '{child_macro}.{child_variable}'"
            ),
            Self::PublicInputAliasMissingParentNetBinding {
                variable,
                child_instance,
                interface_port,
                net,
            } => write!(
                formatter,
                "parent variable '{variable}' has no binding on net '{net}' connected to '{child_instance}.{interface_port}'"
            ),
            Self::UnknownPublicInputAliasPort { child_macro, port } => write!(
                formatter,
                "public input alias references unprojected interface port '{port}' in child macro '{child_macro}'"
            ),
            Self::InvalidCompactSeed { error } => {
                write!(formatter, "invalid compact seed: {error}")
            }
            Self::EmptyDerivationChild { rule_index } => write!(
                formatter,
                "derivation rule {rule_index} has an empty child instance"
            ),
            Self::EmptyDerivationExpression { rule_index } => write!(
                formatter,
                "derivation rule {rule_index} has an empty expression"
            ),
            Self::InvalidDerivationExpression { rule_index, reason } => write!(
                formatter,
                "derivation rule {rule_index} has an invalid expression: {reason}"
            ),
            Self::UnknownDerivationChild {
                rule_index,
                child_instance,
            } => write!(
                formatter,
                "derivation rule {rule_index} references unknown child instance '{child_instance}'"
            ),
            Self::DerivationTargetsNonMacro {
                rule_index,
                child_instance,
            } => write!(
                formatter,
                "derivation rule {rule_index} target '{child_instance}' is not a macro instance"
            ),
            Self::EmptyDerivationTarget { rule_index } => {
                write!(
                    formatter,
                    "derivation rule {rule_index} has an empty target name"
                )
            }
            Self::IncompatibleDerivationTarget { rule_index } => write!(
                formatter,
                "derivation rule {rule_index} reduction is incompatible with its target"
            ),
            Self::UnknownDerivationChildMacro {
                rule_index,
                child_macro,
            } => write!(
                formatter,
                "derivation rule {rule_index} references unregistered child macro '{child_macro}'"
            ),
            Self::UnknownDerivedSpecification {
                rule_index,
                child_macro,
                specification,
            } => write!(
                formatter,
                "derivation rule {rule_index} targets unknown specification '{specification}' in child macro '{child_macro}'"
            ),
            Self::UnknownDerivedDesignVariable {
                rule_index,
                child_macro,
                variable,
            } => write!(
                formatter,
                "derivation rule {rule_index} targets unknown design variable '{variable}' in child macro '{child_macro}'"
            ),
        }
    }
}

/// Resolves derivation targets against the definitions of direct child macros.
pub fn validate_macro_derivation_targets(
    macro_: &Macro,
    macro_catalog: &MacroCatalog,
) -> Vec<MacroExplorationDefinitionError> {
    let mut errors = Vec::new();
    for (rule_index, rule) in macro_.exploration().derivation_rules().iter().enumerate() {
        let Some(instance) = macro_.circuit().instance(rule.child_instance()) else {
            continue;
        };
        let BlockRef::Macro(child_macro_name) = instance.block() else {
            continue;
        };
        let Some(child_macro) = macro_catalog.get(child_macro_name) else {
            errors.push(
                MacroExplorationDefinitionError::UnknownDerivationChildMacro {
                    rule_index,
                    child_macro: child_macro_name.clone(),
                },
            );
            continue;
        };
        match rule.target() {
            MacroDerivationTarget::SpecificationMinimum { specification }
            | MacroDerivationTarget::SpecificationMaximum { specification }
            | MacroDerivationTarget::SpecificationRange { specification } => {
                if child_macro
                    .exploration()
                    .specification(specification)
                    .is_none()
                {
                    errors.push(
                        MacroExplorationDefinitionError::UnknownDerivedSpecification {
                            rule_index,
                            child_macro: child_macro_name.clone(),
                            specification: specification.clone(),
                        },
                    );
                }
            }
            MacroDerivationTarget::DesignVariableRange { variable }
            | MacroDerivationTarget::DesignVariableAllowedValues { variable } => {
                if child_macro
                    .exploration()
                    .design_variable(variable)
                    .is_none()
                {
                    errors.push(
                        MacroExplorationDefinitionError::UnknownDerivedDesignVariable {
                            rule_index,
                            child_macro: child_macro_name.clone(),
                            variable: variable.clone(),
                        },
                    );
                }
            }
        }
    }
    errors
}

/// Resolves parent-owned public input aliases against direct child definitions.
pub fn validate_macro_public_input_aliases(
    macro_: &Macro,
    macro_catalog: &MacroCatalog,
) -> Vec<MacroExplorationDefinitionError> {
    let mut errors = Vec::new();
    for alias in macro_.exploration().public_input_aliases() {
        let Some(instance) = macro_.circuit().instance(alias.child_instance()) else {
            continue;
        };
        let BlockRef::Macro(child_name) = instance.block() else {
            continue;
        };
        let Some(child) = macro_catalog.get(child_name) else {
            errors.push(
                MacroExplorationDefinitionError::UnknownPublicInputAliasChildMacro {
                    child_macro: child_name.clone(),
                },
            );
            continue;
        };
        let Some(parent_variable) = macro_.exploration().design_variable(alias.variable()) else {
            continue;
        };
        let Some(child_variable) = child.exploration().design_variable(alias.child_variable())
        else {
            errors.push(
                MacroExplorationDefinitionError::UnknownPublicInputAliasChildVariable {
                    child_macro: child_name.clone(),
                    variable: alias.child_variable().to_owned(),
                },
            );
            continue;
        };
        if !child_variable.kind().accepts(parent_variable.kind()) {
            errors.push(
                MacroExplorationDefinitionError::PublicInputAliasKindMismatch {
                    variable: alias.variable().to_owned(),
                    child_macro: child_name.clone(),
                    child_variable: alias.child_variable().to_owned(),
                },
            );
        }
        if !parent_variable.bindings().is_empty()
            && let Some(net) = instance.net_for_port(alias.interface_port())
            && !parent_variable.bindings().iter().any(|binding| {
                macro_
                    .circuit()
                    .instance(binding.instance_path())
                    .and_then(|source| source.net_for_port(binding.input()))
                    == Some(net)
            })
        {
            errors.push(
                MacroExplorationDefinitionError::PublicInputAliasMissingParentNetBinding {
                    variable: alias.variable().to_owned(),
                    child_instance: alias.child_instance().to_owned(),
                    interface_port: alias.interface_port().to_owned(),
                    net: net.to_owned(),
                },
            );
        }
        let exposes_port = if child.hierarchy_mode() == MacroHierarchyMode::BlackBox {
            child
                .ports()
                .iter()
                .any(|port| port.name() == alias.interface_port())
        } else {
            child
                .exploration()
                .interface_bindings()
                .iter()
                .any(|binding| binding.port() == alias.interface_port())
        };
        if !exposes_port {
            errors.push(
                MacroExplorationDefinitionError::UnknownPublicInputAliasPort {
                    child_macro: child_name.clone(),
                    port: alias.interface_port().to_owned(),
                },
            );
        }
    }
    errors
}

impl Error for MacroExplorationDefinitionError {}

/// Validates primitive defaults and design variables against the implementation.
pub fn validate_macro_exploration_definition(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
) -> Vec<MacroExplorationDefinitionError> {
    let mut errors = Vec::new();
    let mut default_instances = HashSet::new();
    for (index, default) in macro_.exploration().primitive_defaults().iter().enumerate() {
        let instance_path = default.instance_path();
        if instance_path.trim().is_empty() {
            errors.push(MacroExplorationDefinitionError::EmptyPrimitiveDefaultInstance { index });
            continue;
        }
        if !default_instances.insert(instance_path) {
            errors.push(MacroExplorationDefinitionError::DuplicatePrimitiveDefault {
                instance_path: instance_path.to_owned(),
            });
            continue;
        }
        validate_default(macro_, primitive_catalog, default, &mut errors);
    }

    let mut variable_names = HashSet::new();
    let mut bound_inputs = HashSet::new();
    for (index, variable) in macro_.exploration().design_variables().iter().enumerate() {
        if variable.name().trim().is_empty() {
            errors.push(MacroExplorationDefinitionError::EmptyDesignVariable { index });
        } else if !variable_names.insert(variable.name()) {
            errors.push(MacroExplorationDefinitionError::DuplicateDesignVariable {
                variable: variable.name().to_owned(),
            });
        }
        if variable.bindings().is_empty()
            && !macro_
                .exploration()
                .public_input_aliases()
                .iter()
                .any(|alias| alias.variable() == variable.name())
        {
            errors.push(
                MacroExplorationDefinitionError::EmptyDesignVariableBindings {
                    variable: variable.name().to_owned(),
                },
            );
        }
        for (binding_index, binding) in variable.bindings().iter().enumerate() {
            if binding.instance_path().trim().is_empty() {
                errors.push(
                    MacroExplorationDefinitionError::EmptyDesignVariableBinding {
                        variable: variable.name().to_owned(),
                        binding_index,
                        field: "instance path",
                    },
                );
                continue;
            }
            if binding.input().trim().is_empty() {
                errors.push(
                    MacroExplorationDefinitionError::EmptyDesignVariableBinding {
                        variable: variable.name().to_owned(),
                        binding_index,
                        field: "input name",
                    },
                );
                continue;
            }
            if !bound_inputs.insert((binding.instance_path(), binding.input())) {
                errors.push(
                    MacroExplorationDefinitionError::DuplicateDesignVariableBinding {
                        instance_path: binding.instance_path().to_owned(),
                        input: binding.input().to_owned(),
                    },
                );
                continue;
            }
            validate_variable_binding(macro_, primitive_catalog, variable, binding, &mut errors);
        }
    }
    let mut aliases = HashSet::new();
    for (alias_index, alias) in macro_
        .exploration()
        .public_input_aliases()
        .iter()
        .enumerate()
    {
        for (field, value) in [
            ("parent variable", alias.variable()),
            ("child instance", alias.child_instance()),
            ("child variable", alias.child_variable()),
            ("interface port", alias.interface_port()),
        ] {
            if value.trim().is_empty() {
                errors.push(MacroExplorationDefinitionError::EmptyPublicInputAlias {
                    alias_index,
                    field,
                });
            }
        }
        if !aliases.insert((alias.child_instance(), alias.child_variable())) {
            errors.push(MacroExplorationDefinitionError::DuplicatePublicInputAlias {
                child_instance: alias.child_instance().to_owned(),
                child_variable: alias.child_variable().to_owned(),
            });
        }
        if macro_
            .exploration()
            .design_variable(alias.variable())
            .is_none()
        {
            errors.push(
                MacroExplorationDefinitionError::UnknownPublicInputAliasVariable {
                    variable: alias.variable().to_owned(),
                },
            );
        }
        match macro_.circuit().instance(alias.child_instance()) {
            None => errors.push(
                MacroExplorationDefinitionError::UnknownPublicInputAliasInstance {
                    child_instance: alias.child_instance().to_owned(),
                },
            ),
            Some(instance) if !matches!(instance.block(), BlockRef::Macro(_)) => errors.push(
                MacroExplorationDefinitionError::PublicInputAliasTargetsNonMacro {
                    child_instance: alias.child_instance().to_owned(),
                },
            ),
            Some(_) => {}
        }
    }
    if let Some(seeds) = macro_.exploration().compact_seeds() {
        errors.extend(
            super::seed::validate_compact_seeds(macro_, seeds)
                .into_iter()
                .map(|error| MacroExplorationDefinitionError::InvalidCompactSeed { error }),
        );
    }
    for (rule_index, rule) in macro_.exploration().derivation_rules().iter().enumerate() {
        if rule.child_instance().trim().is_empty() {
            errors.push(MacroExplorationDefinitionError::EmptyDerivationChild { rule_index });
        } else {
            match macro_.circuit().instance(rule.child_instance()) {
                None => errors.push(MacroExplorationDefinitionError::UnknownDerivationChild {
                    rule_index,
                    child_instance: rule.child_instance().to_owned(),
                }),
                Some(instance) if !matches!(instance.block(), BlockRef::Macro(_)) => {
                    errors.push(MacroExplorationDefinitionError::DerivationTargetsNonMacro {
                        rule_index,
                        child_instance: rule.child_instance().to_owned(),
                    })
                }
                Some(_) => {}
            }
        }
        if rule.expression().trim().is_empty() {
            errors.push(MacroExplorationDefinitionError::EmptyDerivationExpression { rule_index });
        } else if let Err(reason) =
            super::specification::validate_numeric_expression(rule.expression())
        {
            errors.push(
                MacroExplorationDefinitionError::InvalidDerivationExpression { rule_index, reason },
            );
        }
        if derivation_target_name(rule.target()).trim().is_empty() {
            errors.push(MacroExplorationDefinitionError::EmptyDerivationTarget { rule_index });
        }
        if !derivation_shapes_are_compatible(rule.reduction(), rule.target()) {
            errors
                .push(MacroExplorationDefinitionError::IncompatibleDerivationTarget { rule_index });
        }
    }
    errors
}

fn derivation_target_name(target: &MacroDerivationTarget) -> &str {
    match target {
        MacroDerivationTarget::SpecificationMinimum { specification }
        | MacroDerivationTarget::SpecificationMaximum { specification }
        | MacroDerivationTarget::SpecificationRange { specification } => specification,
        MacroDerivationTarget::DesignVariableRange { variable }
        | MacroDerivationTarget::DesignVariableAllowedValues { variable } => variable,
    }
}

fn derivation_shapes_are_compatible(
    reduction: MacroDerivationReduction,
    target: &MacroDerivationTarget,
) -> bool {
    match reduction {
        MacroDerivationReduction::Minimum | MacroDerivationReduction::Maximum => matches!(
            target,
            MacroDerivationTarget::SpecificationMinimum { .. }
                | MacroDerivationTarget::SpecificationMaximum { .. }
        ),
        MacroDerivationReduction::Range => matches!(
            target,
            MacroDerivationTarget::SpecificationRange { .. }
                | MacroDerivationTarget::DesignVariableRange { .. }
        ),
        MacroDerivationReduction::UniqueValues => matches!(
            target,
            MacroDerivationTarget::DesignVariableAllowedValues { .. }
        ),
    }
}

fn validate_default(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    default: &super::MacroPrimitiveDefault,
    errors: &mut Vec<MacroExplorationDefinitionError>,
) {
    let Some(instance) = macro_.circuit().instance(default.instance_path()) else {
        errors.push(
            MacroExplorationDefinitionError::UnknownPrimitiveDefaultInstance {
                instance_path: default.instance_path().to_owned(),
            },
        );
        return;
    };
    let BlockRef::Primitive(primitive_name) = instance.block() else {
        errors.push(
            MacroExplorationDefinitionError::PrimitiveDefaultTargetsNonPrimitive {
                instance_path: default.instance_path().to_owned(),
            },
        );
        return;
    };
    let Some(build_spec) = primitive_catalog
        .get(primitive_name)
        .and_then(|primitive| primitive.build.as_ref())
    else {
        return;
    };
    for (input, value) in &default.build_input().values {
        let Some(input_spec) = build_spec.inputs.iter().find(|spec| spec.name == *input) else {
            errors.push(
                MacroExplorationDefinitionError::UnknownPrimitiveDefaultInput {
                    instance_path: default.instance_path().to_owned(),
                    input: input.clone(),
                },
            );
            continue;
        };
        if !input_spec.kind.accepts(value.kind()) {
            errors.push(
                MacroExplorationDefinitionError::PrimitiveDefaultInputKindMismatch {
                    instance_path: default.instance_path().to_owned(),
                    input: input.clone(),
                    expected: input_spec.kind,
                    actual: value.kind(),
                },
            );
        }
        if !value.is_finite() {
            errors.push(
                MacroExplorationDefinitionError::NonFinitePrimitiveDefaultInput {
                    instance_path: default.instance_path().to_owned(),
                    input: input.clone(),
                },
            );
        }
    }
}

fn validate_variable_binding(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    variable: &super::MacroDesignVariable,
    binding: &super::MacroDesignVariableBinding,
    errors: &mut Vec<MacroExplorationDefinitionError>,
) {
    let Some(instance) = macro_.circuit().instance(binding.instance_path()) else {
        errors.push(
            MacroExplorationDefinitionError::UnknownDesignVariableInstance {
                variable: variable.name().to_owned(),
                instance_path: binding.instance_path().to_owned(),
            },
        );
        return;
    };
    let BlockRef::Primitive(primitive_name) = instance.block() else {
        errors.push(
            MacroExplorationDefinitionError::DesignVariableTargetsNonPrimitive {
                variable: variable.name().to_owned(),
                instance_path: binding.instance_path().to_owned(),
            },
        );
        return;
    };
    let Some(build_spec) = primitive_catalog
        .get(primitive_name)
        .and_then(|primitive| primitive.build.as_ref())
    else {
        return;
    };
    let Some(input_spec) = build_spec
        .inputs
        .iter()
        .find(|input| input.name == binding.input())
    else {
        errors.push(
            MacroExplorationDefinitionError::UnknownDesignVariableInput {
                variable: variable.name().to_owned(),
                instance_path: binding.instance_path().to_owned(),
                input: binding.input().to_owned(),
            },
        );
        return;
    };
    if !input_spec.kind.accepts(variable.kind()) {
        errors.push(
            MacroExplorationDefinitionError::DesignVariableInputKindMismatch {
                variable: variable.name().to_owned(),
                instance_path: binding.instance_path().to_owned(),
                input: binding.input().to_owned(),
                expected: input_spec.kind,
                actual: variable.kind(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::circuit::Circuit;
    use crate::primitive::build::{
        PrimitiveBuildInput, PrimitiveBuildInputKind, PrimitiveBuildInputSpec, PrimitiveBuildSpec,
        PrimitiveBuildValue, SweepMode,
    };
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};

    use super::*;
    use crate::macro_model::{
        MacroDerivationReduction, MacroDerivationRule, MacroDerivationTarget, MacroDesignVariable,
        MacroDesignVariableBinding, MacroInterfaceBinding, MacroOutputSource,
        MacroPrimitiveDefault, MacroPublicInputAlias, MacroSpecification, MacroSpecificationBounds,
        MacroSpecificationSource,
    };

    fn primitive_catalog() -> PrimitiveCatalog {
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(PrimitiveManifest {
            name: "device".to_owned(),
            version: "1".to_owned(),
            description: None,
            subckt_name: "device".to_owned(),
            pins: vec![Pin {
                name: "OUT".to_owned(),
                role: PinRole::Output,
            }],
            files: PrimitiveFiles {
                netlist: "device.spice".to_owned(),
                build: None,
                symbol: None,
            },
            small_signal: None,
            physical_model: None,
            transistor_type: Some("nmos".to_owned()),
            layout_params: None,
            lut_config: None,
            build: Some(PrimitiveBuildSpec {
                inputs: vec![PrimitiveBuildInputSpec {
                    name: "current".to_owned(),
                    kind: PrimitiveBuildInputKind::Scalar,
                    required: true,
                    source: None,
                }],
                sweep_mode: SweepMode::Aligned,
                derived: Vec::new(),
                lut: Vec::new(),
                columns: Vec::new(),
            }),
        });
        catalog
    }

    fn macro_() -> Macro {
        Macro::new(
            "stage",
            Vec::new(),
            Circuit::builder()
                .primitive("x1", "device", [("OUT", "OUT")])
                .build(),
            Circuit::builder().resistor("r", "OUT", "0", 1.0).build(),
        )
    }

    #[test]
    fn validates_defaults_and_public_variable_bindings() {
        let valid = macro_()
            .with_primitive_default(MacroPrimitiveDefault::new(
                "x1",
                PrimitiveBuildInput::new(HashMap::from([(
                    "current".to_owned(),
                    PrimitiveBuildValue::Scalar(1.0),
                )])),
                Vec::new(),
            ))
            .with_design_variable(MacroDesignVariable::new(
                "current",
                PrimitiveBuildInputKind::Scalar,
                vec![MacroDesignVariableBinding::new("x1", "current")],
            ));
        assert!(validate_macro_exploration_definition(&valid, &primitive_catalog()).is_empty());

        let invalid = macro_()
            .with_primitive_default(MacroPrimitiveDefault::new(
                "x1",
                PrimitiveBuildInput::new(HashMap::from([(
                    "missing".to_owned(),
                    PrimitiveBuildValue::Scalar(1.0),
                )])),
                Vec::new(),
            ))
            .with_design_variable(MacroDesignVariable::new(
                "bad",
                PrimitiveBuildInputKind::Vector,
                vec![
                    MacroDesignVariableBinding::new("x1", "current"),
                    MacroDesignVariableBinding::new("x1", "current"),
                ],
            ));
        let errors = validate_macro_exploration_definition(&invalid, &primitive_catalog());
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroExplorationDefinitionError::UnknownPrimitiveDefaultInput { input, .. }
                if input == "missing"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroExplorationDefinitionError::DesignVariableInputKindMismatch { .. }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroExplorationDefinitionError::DuplicateDesignVariableBinding { .. }
        )));
    }

    #[test]
    fn validates_derivation_shapes_and_resolves_public_child_targets() {
        let child = Macro::new(
            "child",
            Vec::new(),
            Circuit::default(),
            Circuit::builder().resistor("r", "OUT", "0", 1.0).build(),
        )
        .with_specification(MacroSpecification::new(
            "gain",
            MacroSpecificationSource::expression("1"),
            MacroSpecificationBounds::unbounded(),
        ))
        .with_design_variable(MacroDesignVariable::new(
            "biases",
            PrimitiveBuildInputKind::Vector,
            vec![MacroDesignVariableBinding::new("x1", "current")],
        ));
        let parent = Macro::new(
            "parent",
            Vec::new(),
            Circuit::builder()
                .macro_instance("xchild", "child", std::iter::empty::<(&str, &str)>())
                .build(),
            Circuit::builder().resistor("r", "OUT", "0", 1.0).build(),
        )
        .with_derivation_rule(MacroDerivationRule::new(
            "xchild",
            "parent_gain / 2",
            MacroDerivationReduction::Minimum,
            MacroDerivationTarget::specification_minimum("gain"),
        ))
        .with_derivation_rule(MacroDerivationRule::new(
            "xchild",
            "vout",
            MacroDerivationReduction::UniqueValues,
            MacroDerivationTarget::design_variable_allowed_values("biases"),
        ));
        let catalog = MacroCatalog::from_macros([child]).unwrap();

        assert!(validate_macro_exploration_definition(&parent, &primitive_catalog()).is_empty());
        assert!(validate_macro_derivation_targets(&parent, &catalog).is_empty());

        let invalid = parent.with_derivation_rule(MacroDerivationRule::new(
            "xchild",
            "vout",
            MacroDerivationReduction::UniqueValues,
            MacroDerivationTarget::specification_range("gain"),
        ));
        assert!(
            validate_macro_exploration_definition(&invalid, &primitive_catalog())
                .iter()
                .any(|error| matches!(
                    error,
                    MacroExplorationDefinitionError::IncompatibleDerivationTarget { rule_index: 2 }
                ))
        );
    }

    #[test]
    fn accepts_only_shape_compatible_reduction_targets() {
        let reductions = [
            MacroDerivationReduction::Minimum,
            MacroDerivationReduction::Maximum,
            MacroDerivationReduction::Range,
            MacroDerivationReduction::UniqueValues,
        ];
        let targets = [
            MacroDerivationTarget::specification_minimum("x"),
            MacroDerivationTarget::specification_maximum("x"),
            MacroDerivationTarget::specification_range("x"),
            MacroDerivationTarget::design_variable_range("x"),
            MacroDerivationTarget::design_variable_allowed_values("x"),
        ];
        let compatible = reductions
            .into_iter()
            .flat_map(|reduction| {
                targets
                    .iter()
                    .map(move |target| derivation_shapes_are_compatible(reduction, target))
            })
            .filter(|compatible| *compatible)
            .count();

        assert_eq!(compatible, 7);
        assert!(derivation_shapes_are_compatible(
            MacroDerivationReduction::Maximum,
            &MacroDerivationTarget::specification_minimum("x"),
        ));
        assert!(!derivation_shapes_are_compatible(
            MacroDerivationReduction::UniqueValues,
            &MacroDerivationTarget::design_variable_range("x"),
        ));
    }

    #[test]
    fn validates_public_input_aliases_against_the_child_api() {
        let child = Macro::new("child", Vec::new(), Circuit::default(), Circuit::default())
            .with_design_variable(MacroDesignVariable::new(
                "bias",
                PrimitiveBuildInputKind::Vector,
                vec![MacroDesignVariableBinding::new("x1", "current")],
            ))
            .with_interface_binding(MacroInterfaceBinding::new(
                "OUT",
                MacroOutputSource::candidate_column("x1", "out"),
            ));
        let catalog = MacroCatalog::from_macros([child]).unwrap();
        let parent = Macro::new(
            "parent",
            Vec::new(),
            Circuit::builder()
                .macro_instance("xchild", "child", std::iter::empty::<(&str, &str)>())
                .build(),
            Circuit::default(),
        )
        .with_design_variable(MacroDesignVariable::new(
            "node_voltage",
            PrimitiveBuildInputKind::Vector,
            Vec::new(),
        ))
        .with_public_input_alias(MacroPublicInputAlias::new(
            "node_voltage",
            "xchild",
            "bias",
            "OUT",
        ));

        assert!(validate_macro_exploration_definition(&parent, &primitive_catalog()).is_empty());
        assert!(validate_macro_public_input_aliases(&parent, &catalog).is_empty());

        let invalid = parent.with_public_input_alias(MacroPublicInputAlias::new(
            "node_voltage",
            "xchild",
            "missing",
            "OTHER",
        ));
        let errors = validate_macro_public_input_aliases(&invalid, &catalog);
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroExplorationDefinitionError::UnknownPublicInputAliasChildVariable { variable, .. }
                if variable == "missing"
        )));
    }
}
