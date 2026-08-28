//! Validation of definition-owned primitive defaults and public design variables.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::BlockRef;
use crate::primitive::build::PrimitiveBuildInputKind;

use super::Macro;

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
        }
    }
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
        if variable.bindings().is_empty() {
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
    errors
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
        MacroDesignVariable, MacroDesignVariableBinding, MacroPrimitiveDefault,
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
}
