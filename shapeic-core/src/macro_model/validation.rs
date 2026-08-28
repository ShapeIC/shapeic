use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::{BlockRef, Circuit, CircuitValue, LinearElement};

use super::specification::PreparedMacroSpecifications;
use super::{
    Macro, MacroCatalog, MacroExplorationDefinitionError, MacroOutputSource,
    MacroSpecificationEvaluationError, MacroTestbenchSource,
    validate_macro_derivation_targets, validate_macro_exploration_definition,
};

/// Identifies which circuit of a macro contains a validation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroCircuitKind {
    /// The circuit explored to implement the macro.
    Implementation,
    /// The linear circuit exposed to parent macros.
    CompactModel,
}

impl fmt::Display for MacroCircuitKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Implementation => formatter.write_str("implementation"),
            Self::CompactModel => formatter.write_str("compact model"),
        }
    }
}

/// One structural error found while validating a macro definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroValidationError {
    EmptyMacroName,
    EmptyPortName {
        macro_name: String,
        port_index: usize,
    },
    DuplicatePort {
        macro_name: String,
        port: String,
    },
    EmptyCircuit {
        macro_name: String,
        circuit: MacroCircuitKind,
    },
    MissingPublicPortNet {
        macro_name: String,
        port: String,
    },
    EmptyInstanceName {
        macro_name: String,
        circuit: MacroCircuitKind,
        instance_index: usize,
    },
    DuplicateInstance {
        macro_name: String,
        circuit: MacroCircuitKind,
        instance: String,
    },
    EmptyConnectionPort {
        macro_name: String,
        circuit: MacroCircuitKind,
        instance: String,
        connection_index: usize,
    },
    EmptyNet {
        macro_name: String,
        circuit: MacroCircuitKind,
        instance: String,
        port: String,
    },
    DuplicateConnection {
        macro_name: String,
        circuit: MacroCircuitKind,
        instance: String,
        port: String,
    },
    UnknownPrimitive {
        macro_name: String,
        instance: String,
        primitive: String,
    },
    MissingPrimitiveSmallSignalModel {
        macro_name: String,
        instance: String,
        primitive: String,
    },
    EmptySmallSignalModel {
        macro_name: String,
        instance: String,
        primitive: String,
    },
    EmptySmallSignalBranchName {
        macro_name: String,
        instance: String,
        primitive: String,
        branch_index: usize,
    },
    DuplicateSmallSignalBranch {
        macro_name: String,
        instance: String,
        primitive: String,
        branch: String,
    },
    EmptySmallSignalBranchPin {
        macro_name: String,
        instance: String,
        primitive: String,
        branch: String,
        terminal: &'static str,
    },
    UnknownSmallSignalBranchPin {
        macro_name: String,
        instance: String,
        primitive: String,
        branch: String,
        terminal: &'static str,
        pin: String,
    },
    EmptyPhysicalLutPrimitive {
        macro_name: String,
        instance: String,
        primitive: String,
    },
    EmptyPhysicalOperatingPointBranch {
        macro_name: String,
        instance: String,
        primitive: String,
    },
    UnknownPhysicalOperatingPointBranch {
        macro_name: String,
        instance: String,
        primitive: String,
        branch: String,
    },
    EmptyPhysicalPortMap {
        macro_name: String,
        instance: String,
        primitive: String,
    },
    EmptyPhysicalPort {
        macro_name: String,
        instance: String,
        primitive: String,
        mapping_index: usize,
    },
    DuplicatePhysicalPort {
        macro_name: String,
        instance: String,
        primitive: String,
        port: String,
    },
    EmptyPhysicalPin {
        macro_name: String,
        instance: String,
        primitive: String,
        port: String,
    },
    UnknownPhysicalPin {
        macro_name: String,
        instance: String,
        primitive: String,
        port: String,
        pin: String,
    },
    MissingPhysicalPinMapping {
        macro_name: String,
        instance: String,
        primitive: String,
        pin: String,
    },
    UnknownMacro {
        macro_name: String,
        instance: String,
        referenced_macro: String,
    },
    UnknownBlockPort {
        macro_name: String,
        circuit: MacroCircuitKind,
        instance: String,
        block: String,
        port: String,
    },
    MissingBlockConnection {
        macro_name: String,
        circuit: MacroCircuitKind,
        instance: String,
        block: String,
        port: String,
    },
    NonLinearCompactBlock {
        macro_name: String,
        instance: String,
        block: String,
    },
    EmptyParameter {
        macro_name: String,
        circuit: MacroCircuitKind,
        instance: String,
    },
    NonFiniteConstant {
        macro_name: String,
        circuit: MacroCircuitKind,
        instance: String,
    },
    EmptyTestbenchName {
        macro_name: String,
        testbench_index: usize,
    },
    DuplicateTestbench {
        macro_name: String,
        testbench: String,
    },
    EmptyTestbenchSource {
        macro_name: String,
        testbench: String,
    },
    EmptyTransferNode {
        macro_name: String,
        testbench: String,
        node: &'static str,
    },
    InvalidAcConfig {
        macro_name: String,
        testbench: String,
    },
    InvalidAcPolicy {
        macro_name: String,
        testbench: String,
    },
    EmptyCompactOutputParameter {
        macro_name: String,
    },
    DuplicateCompactOutputParameter {
        macro_name: String,
        parameter: String,
    },
    UnknownCompactOutputParameter {
        macro_name: String,
        parameter: String,
    },
    MissingCompactOutputBinding {
        macro_name: String,
        parameter: String,
    },
    EmptyInterfacePort {
        macro_name: String,
    },
    DuplicateInterfacePort {
        macro_name: String,
        port: String,
    },
    UnknownInterfacePort {
        macro_name: String,
        port: String,
    },
    InvalidOutputSource {
        macro_name: String,
        target: String,
        reason: String,
    },
    InvalidSpecification {
        macro_name: String,
        error: MacroSpecificationEvaluationError,
    },
    InvalidExplorationDefinition {
        macro_name: String,
        error: MacroExplorationDefinitionError,
    },
    CyclicDependency {
        path: Vec<String>,
    },
}

impl fmt::Display for MacroValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMacroName => formatter.write_str("macro name is empty"),
            Self::EmptyPortName {
                macro_name,
                port_index,
            } => write!(
                formatter,
                "macro '{macro_name}' has an empty port at index {port_index}"
            ),
            Self::DuplicatePort { macro_name, port } => {
                write!(
                    formatter,
                    "macro '{macro_name}' declares port '{port}' more than once"
                )
            }
            Self::EmptyCircuit {
                macro_name,
                circuit,
            } => write!(formatter, "macro '{macro_name}' has an empty {circuit}"),
            Self::MissingPublicPortNet { macro_name, port } => write!(
                formatter,
                "macro '{macro_name}' implementation does not use public port net '{port}'"
            ),
            Self::EmptyInstanceName {
                macro_name,
                circuit,
                instance_index,
            } => write!(
                formatter,
                "macro '{macro_name}' {circuit} has an empty instance name at index {instance_index}"
            ),
            Self::DuplicateInstance {
                macro_name,
                circuit,
                instance,
            } => write!(
                formatter,
                "macro '{macro_name}' {circuit} declares instance '{instance}' more than once"
            ),
            Self::EmptyConnectionPort {
                macro_name,
                circuit,
                instance,
                connection_index,
            } => write!(
                formatter,
                "macro '{macro_name}' {circuit} instance '{instance}' has an empty port at connection {connection_index}"
            ),
            Self::EmptyNet {
                macro_name,
                circuit,
                instance,
                port,
            } => write!(
                formatter,
                "macro '{macro_name}' {circuit} instance '{instance}' connects port '{port}' to an empty net"
            ),
            Self::DuplicateConnection {
                macro_name,
                circuit,
                instance,
                port,
            } => write!(
                formatter,
                "macro '{macro_name}' {circuit} instance '{instance}' connects port '{port}' more than once"
            ),
            Self::UnknownPrimitive {
                macro_name,
                instance,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' references unknown primitive '{primitive}'"
            ),
            Self::MissingPrimitiveSmallSignalModel {
                macro_name,
                instance,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' has no small-signal model"
            ),
            Self::EmptySmallSignalModel {
                macro_name,
                instance,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' has an empty small-signal model"
            ),
            Self::EmptySmallSignalBranchName {
                macro_name,
                instance,
                primitive,
                branch_index,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' has an empty small-signal branch name at index {branch_index}"
            ),
            Self::DuplicateSmallSignalBranch {
                macro_name,
                instance,
                primitive,
                branch,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' declares small-signal branch '{branch}' more than once"
            ),
            Self::EmptySmallSignalBranchPin {
                macro_name,
                instance,
                primitive,
                branch,
                terminal,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' branch '{branch}' has an empty {terminal} pin"
            ),
            Self::UnknownSmallSignalBranchPin {
                macro_name,
                instance,
                primitive,
                branch,
                terminal,
                pin,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' branch '{branch}' references unknown {terminal} pin '{pin}'"
            ),
            Self::EmptyPhysicalLutPrimitive {
                macro_name,
                instance,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' has an empty physical LUT primitive"
            ),
            Self::EmptyPhysicalOperatingPointBranch {
                macro_name,
                instance,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' has an empty physical operating-point branch"
            ),
            Self::UnknownPhysicalOperatingPointBranch {
                macro_name,
                instance,
                primitive,
                branch,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' physical model references unknown branch '{branch}'"
            ),
            Self::EmptyPhysicalPortMap {
                macro_name,
                instance,
                primitive,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' has no physical port mappings"
            ),
            Self::EmptyPhysicalPort {
                macro_name,
                instance,
                primitive,
                mapping_index,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' has an empty physical port at mapping {mapping_index}"
            ),
            Self::DuplicatePhysicalPort {
                macro_name,
                instance,
                primitive,
                port,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' maps physical port '{port}' more than once"
            ),
            Self::EmptyPhysicalPin {
                macro_name,
                instance,
                primitive,
                port,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' maps physical port '{port}' to an empty pin"
            ),
            Self::UnknownPhysicalPin {
                macro_name,
                instance,
                primitive,
                port,
                pin,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' maps physical port '{port}' to unknown pin '{pin}'"
            ),
            Self::MissingPhysicalPinMapping {
                macro_name,
                instance,
                primitive,
                pin,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' primitive '{primitive}' physical model does not map pin '{pin}'"
            ),
            Self::UnknownMacro {
                macro_name,
                instance,
                referenced_macro,
            } => write!(
                formatter,
                "macro '{macro_name}' instance '{instance}' references unknown macro '{referenced_macro}'"
            ),
            Self::UnknownBlockPort {
                macro_name,
                circuit,
                instance,
                block,
                port,
            } => write!(
                formatter,
                "macro '{macro_name}' {circuit} instance '{instance}' connects unknown port '{port}' on '{block}'"
            ),
            Self::MissingBlockConnection {
                macro_name,
                circuit,
                instance,
                block,
                port,
            } => write!(
                formatter,
                "macro '{macro_name}' {circuit} instance '{instance}' is missing port '{port}' on '{block}'"
            ),
            Self::NonLinearCompactBlock {
                macro_name,
                instance,
                block,
            } => write!(
                formatter,
                "macro '{macro_name}' compact model instance '{instance}' references non-linear block '{block}'"
            ),
            Self::EmptyParameter {
                macro_name,
                circuit,
                instance,
            } => write!(
                formatter,
                "macro '{macro_name}' {circuit} instance '{instance}' has an empty parameter name"
            ),
            Self::NonFiniteConstant {
                macro_name,
                circuit,
                instance,
            } => write!(
                formatter,
                "macro '{macro_name}' {circuit} instance '{instance}' has a non-finite value"
            ),
            Self::EmptyTestbenchName {
                macro_name,
                testbench_index,
            } => write!(
                formatter,
                "macro '{macro_name}' has an empty testbench name at index {testbench_index}"
            ),
            Self::DuplicateTestbench {
                macro_name,
                testbench,
            } => write!(
                formatter,
                "macro '{macro_name}' declares testbench '{testbench}' more than once"
            ),
            Self::EmptyTestbenchSource {
                macro_name,
                testbench,
            } => write!(
                formatter,
                "macro '{macro_name}' testbench '{testbench}' has an empty source"
            ),
            Self::EmptyTransferNode {
                macro_name,
                testbench,
                node,
            } => write!(
                formatter,
                "macro '{macro_name}' testbench '{testbench}' has an empty transfer {node} node"
            ),
            Self::InvalidAcConfig {
                macro_name,
                testbench,
            } => write!(
                formatter,
                "macro '{macro_name}' testbench '{testbench}' has an invalid AC configuration"
            ),
            Self::InvalidAcPolicy {
                macro_name,
                testbench,
            } => write!(
                formatter,
                "macro '{macro_name}' testbench '{testbench}' has an invalid AC policy"
            ),
            Self::EmptyCompactOutputParameter { macro_name } => write!(
                formatter,
                "macro '{macro_name}' has a compact output with an empty parameter"
            ),
            Self::DuplicateCompactOutputParameter {
                macro_name,
                parameter,
            } => write!(
                formatter,
                "macro '{macro_name}' binds compact parameter '{parameter}' more than once"
            ),
            Self::UnknownCompactOutputParameter {
                macro_name,
                parameter,
            } => write!(
                formatter,
                "macro '{macro_name}' binds undeclared compact parameter '{parameter}'"
            ),
            Self::MissingCompactOutputBinding {
                macro_name,
                parameter,
            } => write!(
                formatter,
                "macro '{macro_name}' compact parameter '{parameter}' has no output binding"
            ),
            Self::EmptyInterfacePort { macro_name } => write!(
                formatter,
                "macro '{macro_name}' has an interface binding with an empty port"
            ),
            Self::DuplicateInterfacePort { macro_name, port } => write!(
                formatter,
                "macro '{macro_name}' binds interface port '{port}' more than once"
            ),
            Self::UnknownInterfacePort { macro_name, port } => write!(
                formatter,
                "macro '{macro_name}' binds unknown interface port '{port}'"
            ),
            Self::InvalidOutputSource {
                macro_name,
                target,
                reason,
            } => write!(
                formatter,
                "macro '{macro_name}' output binding '{target}' has an invalid source: {reason}"
            ),
            Self::InvalidSpecification { macro_name, error } => write!(
                formatter,
                "macro '{macro_name}' has an invalid specification: {error}"
            ),
            Self::InvalidExplorationDefinition { macro_name, error } => write!(
                formatter,
                "macro '{macro_name}' has an invalid exploration definition: {error}"
            ),
            Self::CyclicDependency { path } => {
                write!(formatter, "cyclic macro dependency: {}", path.join(" -> "))
            }
        }
    }
}

impl Error for MacroValidationError {}

/// Validates one macro and returns every independent structural error found.
pub fn validate_macro(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
) -> Vec<MacroValidationError> {
    let mut errors = Vec::new();
    let macro_name = macro_.name().to_owned();

    if macro_.name().trim().is_empty() {
        errors.push(MacroValidationError::EmptyMacroName);
    }

    let mut port_names = HashSet::new();
    for (port_index, port) in macro_.ports().iter().enumerate() {
        if port.name().trim().is_empty() {
            errors.push(MacroValidationError::EmptyPortName {
                macro_name: macro_name.clone(),
                port_index,
            });
        } else if !port_names.insert(port.name()) {
            errors.push(MacroValidationError::DuplicatePort {
                macro_name: macro_name.clone(),
                port: port.name().to_owned(),
            });
        }
    }

    validate_circuit(
        macro_,
        macro_.circuit(),
        MacroCircuitKind::Implementation,
        primitive_catalog,
        macro_catalog,
        &mut errors,
    );
    validate_circuit(
        macro_,
        macro_.compact_model(),
        MacroCircuitKind::CompactModel,
        primitive_catalog,
        macro_catalog,
        &mut errors,
    );
    validate_testbenches(macro_, &mut errors);
    if let Err(error) = PreparedMacroSpecifications::new(macro_) {
        errors.push(MacroValidationError::InvalidSpecification {
            macro_name: macro_name.clone(),
            error,
        });
    }
    errors.extend(
        validate_macro_exploration_definition(macro_, primitive_catalog)
            .into_iter()
            .map(|error| MacroValidationError::InvalidExplorationDefinition {
                macro_name: macro_name.clone(),
                error,
            }),
    );
    errors.extend(
        validate_macro_derivation_targets(macro_, macro_catalog)
            .into_iter()
            .map(|error| MacroValidationError::InvalidExplorationDefinition {
                macro_name: macro_name.clone(),
                error,
            }),
    );
    validate_output_bindings(macro_, &mut errors);
    errors
}

/// Validates every registered macro and the dependency graph between them.
pub fn validate_macro_catalog(
    macro_catalog: &MacroCatalog,
    primitive_catalog: &PrimitiveCatalog,
) -> Vec<MacroValidationError> {
    let mut errors = macro_catalog
        .list()
        .into_iter()
        .flat_map(|macro_| validate_macro(macro_, primitive_catalog, macro_catalog))
        .collect::<Vec<_>>();
    validate_dependency_graph(macro_catalog, &mut errors);
    errors
}

fn validate_circuit(
    macro_: &Macro,
    circuit: &Circuit,
    kind: MacroCircuitKind,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
    errors: &mut Vec<MacroValidationError>,
) {
    let macro_name = macro_.name().to_owned();
    if circuit.instances().is_empty() {
        errors.push(MacroValidationError::EmptyCircuit {
            macro_name: macro_name.clone(),
            circuit: kind,
        });
    }

    if kind == MacroCircuitKind::Implementation {
        let used_nets = circuit
            .instances()
            .iter()
            .flat_map(|instance| instance.connections())
            .map(|connection| connection.net())
            .collect::<HashSet<_>>();
        for port in macro_.ports() {
            if !port.name().trim().is_empty() && !used_nets.contains(port.name()) {
                errors.push(MacroValidationError::MissingPublicPortNet {
                    macro_name: macro_name.clone(),
                    port: port.name().to_owned(),
                });
            }
        }
    }

    let mut instance_names = HashSet::new();
    for (instance_index, instance) in circuit.instances().iter().enumerate() {
        if instance.name().trim().is_empty() {
            errors.push(MacroValidationError::EmptyInstanceName {
                macro_name: macro_name.clone(),
                circuit: kind,
                instance_index,
            });
        } else if !instance_names.insert(instance.name()) {
            errors.push(MacroValidationError::DuplicateInstance {
                macro_name: macro_name.clone(),
                circuit: kind,
                instance: instance.name().to_owned(),
            });
        }

        let mut connected_ports = HashSet::new();
        for (connection_index, connection) in instance.connections().iter().enumerate() {
            if connection.port().trim().is_empty() {
                errors.push(MacroValidationError::EmptyConnectionPort {
                    macro_name: macro_name.clone(),
                    circuit: kind,
                    instance: instance.name().to_owned(),
                    connection_index,
                });
            } else if !connected_ports.insert(connection.port()) {
                errors.push(MacroValidationError::DuplicateConnection {
                    macro_name: macro_name.clone(),
                    circuit: kind,
                    instance: instance.name().to_owned(),
                    port: connection.port().to_owned(),
                });
            }
            if connection.net().trim().is_empty() {
                errors.push(MacroValidationError::EmptyNet {
                    macro_name: macro_name.clone(),
                    circuit: kind,
                    instance: instance.name().to_owned(),
                    port: connection.port().to_owned(),
                });
            }
        }

        let expected = match instance.block() {
            BlockRef::Primitive(primitive) if kind == MacroCircuitKind::CompactModel => {
                errors.push(MacroValidationError::NonLinearCompactBlock {
                    macro_name: macro_name.clone(),
                    instance: instance.name().to_owned(),
                    block: primitive.clone(),
                });
                None
            }
            BlockRef::Macro(referenced_macro) if kind == MacroCircuitKind::CompactModel => {
                errors.push(MacroValidationError::NonLinearCompactBlock {
                    macro_name: macro_name.clone(),
                    instance: instance.name().to_owned(),
                    block: referenced_macro.clone(),
                });
                None
            }
            BlockRef::Primitive(primitive) => match primitive_catalog.get(primitive) {
                Some(manifest) => {
                    validate_primitive_small_signal(macro_, instance.name(), manifest, errors);
                    validate_primitive_physical_model(macro_, instance.name(), manifest, errors);
                    Some((
                        primitive.clone(),
                        manifest
                            .pins
                            .iter()
                            .map(|pin| pin.name.as_str())
                            .collect::<Vec<_>>(),
                    ))
                }
                None => {
                    errors.push(MacroValidationError::UnknownPrimitive {
                        macro_name: macro_name.clone(),
                        instance: instance.name().to_owned(),
                        primitive: primitive.clone(),
                    });
                    None
                }
            },
            BlockRef::Macro(referenced_macro) => match macro_catalog.get(referenced_macro) {
                Some(definition) => Some((
                    referenced_macro.clone(),
                    definition
                        .ports()
                        .iter()
                        .map(|port| port.name())
                        .collect::<Vec<_>>(),
                )),
                None => {
                    errors.push(MacroValidationError::UnknownMacro {
                        macro_name: macro_name.clone(),
                        instance: instance.name().to_owned(),
                        referenced_macro: referenced_macro.clone(),
                    });
                    None
                }
            },
            BlockRef::Element(element) => {
                validate_element_value(macro_, kind, instance.name(), element, errors);
                Some((
                    element_name(element).to_owned(),
                    element_ports(element).to_vec(),
                ))
            }
        };

        if let Some((block, expected_ports)) = expected {
            let expected_ports = expected_ports.into_iter().collect::<HashSet<_>>();
            for connection in instance.connections() {
                if !connection.port().trim().is_empty()
                    && !expected_ports.contains(connection.port())
                {
                    errors.push(MacroValidationError::UnknownBlockPort {
                        macro_name: macro_name.clone(),
                        circuit: kind,
                        instance: instance.name().to_owned(),
                        block: block.clone(),
                        port: connection.port().to_owned(),
                    });
                }
            }
            for port in expected_ports {
                if !connected_ports.contains(port) {
                    errors.push(MacroValidationError::MissingBlockConnection {
                        macro_name: macro_name.clone(),
                        circuit: kind,
                        instance: instance.name().to_owned(),
                        block: block.clone(),
                        port: port.to_owned(),
                    });
                }
            }
        }
    }
}

fn validate_primitive_physical_model(
    macro_: &Macro,
    instance: &str,
    primitive: &crate::primitive::manifest::PrimitiveManifest,
    errors: &mut Vec<MacroValidationError>,
) {
    let Some(physical) = primitive.physical_model.as_ref() else {
        return;
    };
    let context = || {
        (
            macro_.name().to_owned(),
            instance.to_owned(),
            primitive.name.clone(),
        )
    };
    if physical.lut_primitive().trim().is_empty() {
        let (macro_name, instance, primitive) = context();
        errors.push(MacroValidationError::EmptyPhysicalLutPrimitive {
            macro_name,
            instance,
            primitive,
        });
    }
    if physical.operating_point_branch().trim().is_empty() {
        let (macro_name, instance, primitive) = context();
        errors.push(MacroValidationError::EmptyPhysicalOperatingPointBranch {
            macro_name,
            instance,
            primitive,
        });
    } else if !primitive.small_signal.as_ref().is_some_and(|model| {
        model.branch(physical.operating_point_branch()).is_some()
    }) || !primitive.build.as_ref().is_some_and(|build| {
        build
            .lut
            .iter()
            .any(|lut| lut.name == physical.operating_point_branch())
    }) {
        let (macro_name, instance, primitive) = context();
        errors.push(MacroValidationError::UnknownPhysicalOperatingPointBranch {
            macro_name,
            instance,
            primitive,
            branch: physical.operating_point_branch().to_owned(),
        });
    }
    if physical.ports().is_empty() {
        let (macro_name, instance, primitive) = context();
        errors.push(MacroValidationError::EmptyPhysicalPortMap {
            macro_name,
            instance,
            primitive,
        });
    }

    let pins = primitive
        .pins
        .iter()
        .map(|pin| pin.name.as_str())
        .collect::<HashSet<_>>();
    let mut physical_ports = HashSet::new();
    let mut mapped_pins = HashSet::new();
    for (mapping_index, mapping) in physical.ports().iter().enumerate() {
        let port = mapping.physical_port();
        let pin = mapping.primitive_pin();
        if port.trim().is_empty() {
            let (macro_name, instance, primitive) = context();
            errors.push(MacroValidationError::EmptyPhysicalPort {
                macro_name,
                instance,
                primitive,
                mapping_index,
            });
        } else if !physical_ports.insert(port) {
            let (macro_name, instance, primitive) = context();
            errors.push(MacroValidationError::DuplicatePhysicalPort {
                macro_name,
                instance,
                primitive,
                port: port.to_owned(),
            });
        }
        if pin.trim().is_empty() {
            let (macro_name, instance, primitive) = context();
            errors.push(MacroValidationError::EmptyPhysicalPin {
                macro_name,
                instance,
                primitive,
                port: port.to_owned(),
            });
        } else if !pins.contains(pin) {
            let (macro_name, instance, primitive) = context();
            errors.push(MacroValidationError::UnknownPhysicalPin {
                macro_name,
                instance,
                primitive,
                port: port.to_owned(),
                pin: pin.to_owned(),
            });
        } else {
            mapped_pins.insert(pin);
        }
    }
    for pin in pins {
        if !mapped_pins.contains(pin) {
            let (macro_name, instance, primitive) = context();
            errors.push(MacroValidationError::MissingPhysicalPinMapping {
                macro_name,
                instance,
                primitive,
                pin: pin.to_owned(),
            });
        }
    }
}

fn validate_primitive_small_signal(
    macro_: &Macro,
    instance: &str,
    primitive: &crate::primitive::manifest::PrimitiveManifest,
    errors: &mut Vec<MacroValidationError>,
) {
    let context = || {
        (
            macro_.name().to_owned(),
            instance.to_owned(),
            primitive.name.clone(),
        )
    };
    let Some(model) = primitive.small_signal.as_ref() else {
        let (macro_name, instance, primitive) = context();
        errors.push(MacroValidationError::MissingPrimitiveSmallSignalModel {
            macro_name,
            instance,
            primitive,
        });
        return;
    };
    if model.branches().is_empty() {
        let (macro_name, instance, primitive) = context();
        errors.push(MacroValidationError::EmptySmallSignalModel {
            macro_name,
            instance,
            primitive,
        });
    }

    let pins = primitive
        .pins
        .iter()
        .map(|pin| pin.name.as_str())
        .collect::<HashSet<_>>();
    let mut branch_names = HashSet::new();
    for (branch_index, branch) in model.branches().iter().enumerate() {
        if branch.name().trim().is_empty() {
            let (macro_name, instance, primitive) = context();
            errors.push(MacroValidationError::EmptySmallSignalBranchName {
                macro_name,
                instance,
                primitive,
                branch_index,
            });
        } else if !branch_names.insert(branch.name()) {
            let (macro_name, instance, primitive) = context();
            errors.push(MacroValidationError::DuplicateSmallSignalBranch {
                macro_name,
                instance,
                primitive,
                branch: branch.name().to_owned(),
            });
        }

        for (terminal, pin) in [
            ("drain", branch.drain_pin()),
            ("gate", branch.gate_pin()),
            ("source", branch.source_pin()),
            ("bulk", branch.bulk_pin()),
        ] {
            if pin.trim().is_empty() {
                let (macro_name, instance, primitive) = context();
                errors.push(MacroValidationError::EmptySmallSignalBranchPin {
                    macro_name,
                    instance,
                    primitive,
                    branch: branch.name().to_owned(),
                    terminal,
                });
            } else if !pins.contains(pin) {
                let (macro_name, instance, primitive) = context();
                errors.push(MacroValidationError::UnknownSmallSignalBranchPin {
                    macro_name,
                    instance,
                    primitive,
                    branch: branch.name().to_owned(),
                    terminal,
                    pin: pin.to_owned(),
                });
            }
        }
    }
}

fn validate_element_value(
    macro_: &Macro,
    circuit: MacroCircuitKind,
    instance: &str,
    element: &LinearElement,
    errors: &mut Vec<MacroValidationError>,
) {
    let value = match element {
        LinearElement::Resistor { resistance } => resistance,
        LinearElement::Capacitor { capacitance } => capacitance,
        LinearElement::CurrentSource { current } => current,
        LinearElement::VoltageSource { voltage } => voltage,
        LinearElement::VoltageControlledCurrentSource { transconductance } => transconductance,
    };
    match value {
        CircuitValue::Constant(value) if !value.is_finite() => {
            errors.push(MacroValidationError::NonFiniteConstant {
                macro_name: macro_.name().to_owned(),
                circuit,
                instance: instance.to_owned(),
            });
        }
        CircuitValue::Parameter(parameter) if parameter.trim().is_empty() => {
            errors.push(MacroValidationError::EmptyParameter {
                macro_name: macro_.name().to_owned(),
                circuit,
                instance: instance.to_owned(),
            });
        }
        _ => {}
    }
}

fn element_name(element: &LinearElement) -> &'static str {
    match element {
        LinearElement::Resistor { .. } => "resistor",
        LinearElement::Capacitor { .. } => "capacitor",
        LinearElement::CurrentSource { .. } => "current source",
        LinearElement::VoltageSource { .. } => "voltage source",
        LinearElement::VoltageControlledCurrentSource { .. } => "VCCS",
    }
}

fn element_ports(element: &LinearElement) -> &'static [&'static str] {
    match element {
        LinearElement::VoltageControlledCurrentSource { .. } => &["p", "n", "cp", "cn"],
        _ => &["p", "n"],
    }
}

fn validate_testbenches(macro_: &Macro, errors: &mut Vec<MacroValidationError>) {
    let macro_name = macro_.name().to_owned();
    let mut names = HashSet::new();
    for (testbench_index, testbench) in macro_.exploration().testbenches().iter().enumerate() {
        if testbench.name().trim().is_empty() {
            errors.push(MacroValidationError::EmptyTestbenchName {
                macro_name: macro_name.clone(),
                testbench_index,
            });
        } else if !names.insert(testbench.name()) {
            errors.push(MacroValidationError::DuplicateTestbench {
                macro_name: macro_name.clone(),
                testbench: testbench.name().to_owned(),
            });
        }

        let empty_source = match testbench.source() {
            MacroTestbenchSource::Spice(source) => source.trim().is_empty(),
            MacroTestbenchSource::SpiceFile(path) => path.as_os_str().is_empty(),
        };
        if empty_source {
            errors.push(MacroValidationError::EmptyTestbenchSource {
                macro_name: macro_name.clone(),
                testbench: testbench.name().to_owned(),
            });
        }

        let transfer = &testbench.analysis().transfer_function;
        if transfer.input_node.trim().is_empty() {
            errors.push(MacroValidationError::EmptyTransferNode {
                macro_name: macro_name.clone(),
                testbench: testbench.name().to_owned(),
                node: "input",
            });
        }
        if transfer.output_node.trim().is_empty() {
            errors.push(MacroValidationError::EmptyTransferNode {
                macro_name: macro_name.clone(),
                testbench: testbench.name().to_owned(),
                node: "output",
            });
        }
        if testbench.analysis().config.validate().is_err() {
            errors.push(MacroValidationError::InvalidAcConfig {
                macro_name: macro_name.clone(),
                testbench: testbench.name().to_owned(),
            });
        }
        if testbench.analysis().policy.validate().is_err() {
            errors.push(MacroValidationError::InvalidAcPolicy {
                macro_name: macro_name.clone(),
                testbench: testbench.name().to_owned(),
            });
        }
    }
}

pub(super) fn validate_output_bindings(macro_: &Macro, errors: &mut Vec<MacroValidationError>) {
    let macro_name = macro_.name().to_owned();
    let compact_parameters = macro_
        .compact_model()
        .instances()
        .iter()
        .filter_map(|instance| match instance.block() {
            BlockRef::Element(element) => match element_value(element) {
                CircuitValue::Parameter(parameter) if !parameter.trim().is_empty() => {
                    Some(parameter.as_str())
                }
                _ => None,
            },
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut bound_parameters = HashSet::new();

    for binding in macro_.exploration().compact_outputs() {
        let parameter = binding.parameter();
        if parameter.trim().is_empty() {
            errors.push(MacroValidationError::EmptyCompactOutputParameter {
                macro_name: macro_name.clone(),
            });
        } else {
            if !bound_parameters.insert(parameter) {
                errors.push(MacroValidationError::DuplicateCompactOutputParameter {
                    macro_name: macro_name.clone(),
                    parameter: parameter.to_owned(),
                });
            }
            if !compact_parameters.contains(parameter) {
                errors.push(MacroValidationError::UnknownCompactOutputParameter {
                    macro_name: macro_name.clone(),
                    parameter: parameter.to_owned(),
                });
            }
        }
        validate_output_source(
            macro_,
            &format!("compact parameter '{parameter}'"),
            binding.source(),
            errors,
        );
    }
    for parameter in compact_parameters {
        if !bound_parameters.contains(parameter) {
            errors.push(MacroValidationError::MissingCompactOutputBinding {
                macro_name: macro_name.clone(),
                parameter: parameter.to_owned(),
            });
        }
    }

    let public_ports = macro_
        .ports()
        .iter()
        .map(|port| port.name())
        .collect::<HashSet<_>>();
    let mut bound_ports = HashSet::new();
    for binding in macro_.exploration().interface_bindings() {
        let port = binding.port();
        if port.trim().is_empty() {
            errors.push(MacroValidationError::EmptyInterfacePort {
                macro_name: macro_name.clone(),
            });
        } else {
            if !bound_ports.insert(port) {
                errors.push(MacroValidationError::DuplicateInterfacePort {
                    macro_name: macro_name.clone(),
                    port: port.to_owned(),
                });
            }
            if !public_ports.contains(port) {
                errors.push(MacroValidationError::UnknownInterfacePort {
                    macro_name: macro_name.clone(),
                    port: port.to_owned(),
                });
            }
        }
        validate_output_source(
            macro_,
            &format!("interface port '{port}'"),
            binding.source(),
            errors,
        );
    }
}

fn validate_output_source(
    macro_: &Macro,
    target: &str,
    source: &MacroOutputSource,
    errors: &mut Vec<MacroValidationError>,
) {
    let reason = match source {
        MacroOutputSource::CandidateColumn {
            instance_path,
            column,
        } => {
            if instance_path.trim().is_empty() {
                Some("candidate instance path is empty".to_owned())
            } else if column.trim().is_empty() {
                Some("candidate column is empty".to_owned())
            } else {
                match macro_.circuit().instance(instance_path) {
                    None => Some(format!(
                        "candidate instance '{instance_path}' is not in the implementation circuit"
                    )),
                    Some(instance) if matches!(instance.block(), BlockRef::Element(_)) => Some(
                        format!("implementation instance '{instance_path}' has no candidates"),
                    ),
                    Some(_) => None,
                }
            }
        }
        MacroOutputSource::AcMetric { testbench, metric } => {
            if testbench.trim().is_empty() {
                Some("AC testbench name is empty".to_owned())
            } else {
                match macro_.exploration().testbench(testbench) {
                    None => Some(format!("AC testbench '{testbench}' is not declared")),
                    Some(definition) if !definition.analysis().policy.metrics.contains(*metric) => {
                        Some(format!(
                            "AC testbench '{testbench}' does not request metric {}",
                            metric.label()
                        ))
                    }
                    Some(_) => None,
                }
            }
        }
    };
    if let Some(reason) = reason {
        errors.push(MacroValidationError::InvalidOutputSource {
            macro_name: macro_.name().to_owned(),
            target: target.to_owned(),
            reason,
        });
    }
}

fn element_value(element: &LinearElement) -> &CircuitValue {
    match element {
        LinearElement::Resistor { resistance } => resistance,
        LinearElement::Capacitor { capacitance } => capacitance,
        LinearElement::CurrentSource { current } => current,
        LinearElement::VoltageSource { voltage } => voltage,
        LinearElement::VoltageControlledCurrentSource { transconductance } => transconductance,
    }
}

fn validate_dependency_graph(macro_catalog: &MacroCatalog, errors: &mut Vec<MacroValidationError>) {
    let mut completed = HashSet::new();
    let mut active = Vec::new();
    for macro_ in macro_catalog.list() {
        visit_macro(
            macro_.name(),
            macro_catalog,
            &mut completed,
            &mut active,
            errors,
        );
    }
}

fn visit_macro(
    name: &str,
    catalog: &MacroCatalog,
    completed: &mut HashSet<String>,
    active: &mut Vec<String>,
    errors: &mut Vec<MacroValidationError>,
) {
    if completed.contains(name) {
        return;
    }
    if let Some(cycle_start) = active.iter().position(|active_name| active_name == name) {
        let mut path = active[cycle_start..].to_vec();
        path.push(name.to_owned());
        errors.push(MacroValidationError::CyclicDependency { path });
        return;
    }

    let Some(macro_) = catalog.get(name) else {
        return;
    };
    active.push(name.to_owned());
    for dependency in
        macro_
            .circuit()
            .instances()
            .iter()
            .filter_map(|instance| match instance.block() {
                BlockRef::Macro(dependency) => Some(dependency.as_str()),
                _ => None,
            })
    {
        if catalog.contains(dependency) {
            visit_macro(dependency, catalog, completed, active, errors);
        }
    }
    active.pop();
    completed.insert(name.to_owned());
}

#[cfg(test)]
mod tests {
    use crate::analysis::{
        AcMetricSet, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets,
    };
    use crate::catalog::primitive_catalog::PrimitiveCatalog;
    use crate::circuit::Circuit;
    use crate::macro_model::{
        Macro, MacroAcTestbench, MacroCatalog, MacroCompactOutputBinding, MacroInterfaceBinding,
        MacroOutputSource, MacroPort, MacroPortRole,
    };
    use crate::netlist::names::{compact_model_param_name, small_signal_param_name};
    use crate::primitive::manifest::{
        Pin, PinRole, PrimitiveFiles, PrimitiveManifest, PrimitivePhysicalModel,
    };
    use crate::primitive::small_signal::{SmallSignalBranch, SmallSignalModel};
    use crate::testbench::{AcAnalysis, TransferFunction};

    use super::{MacroCircuitKind, MacroValidationError, validate_macro_catalog};

    fn primitive_catalog() -> PrimitiveCatalog {
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(PrimitiveManifest {
            name: "stage_primitive".to_owned(),
            version: "1.0".to_owned(),
            description: None,
            subckt_name: "stage_primitive".to_owned(),
            pins: vec![
                Pin {
                    name: "VIN".to_owned(),
                    role: PinRole::Input,
                },
                Pin {
                    name: "VOUT".to_owned(),
                    role: PinRole::Output,
                },
                Pin {
                    name: "VSS".to_owned(),
                    role: PinRole::Supply,
                },
            ],
            files: PrimitiveFiles {
                netlist: "stage.spice".to_owned(),
                build: None,
                symbol: None,
            },
            small_signal: Some(SmallSignalModel::new(vec![SmallSignalBranch::new(
                "m1", "VOUT", "VIN", "VSS", "VSS",
            )])),
            physical_model: None,
            transistor_type: None,
            layout_params: None,
            lut_config: None,
            build: None,
        });
        catalog
    }

    fn ports() -> Vec<MacroPort> {
        vec![
            MacroPort::new("VIN", MacroPortRole::Input),
            MacroPort::new("VOUT", MacroPortRole::Output),
            MacroPort::new("VSS", MacroPortRole::Ground),
        ]
    }

    fn compact_model() -> Circuit {
        Circuit::builder()
            .vccs("gm", "VOUT", "VSS", "VIN", "VSS", "gm_eq")
            .build()
    }

    fn leaf_macro() -> Macro {
        Macro::new(
            "leaf",
            ports(),
            Circuit::builder()
                .primitive(
                    "xcore",
                    "stage_primitive",
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .build(),
            compact_model(),
        )
        .with_compact_output(MacroCompactOutputBinding::new(
            "gm_eq",
            MacroOutputSource::candidate_column(
                "xcore",
                small_signal_param_name("gm", "xcore", "m1"),
            ),
        ))
    }

    #[test]
    fn accepts_a_valid_hierarchical_catalog() {
        let leaf = leaf_macro();
        let parent = Macro::new(
            "parent",
            ports(),
            Circuit::builder()
                .macro_instance(
                    "xleaf",
                    "leaf",
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .build(),
            compact_model(),
        )
        .with_compact_output(MacroCompactOutputBinding::new(
            "gm_eq",
            MacroOutputSource::candidate_column(
                "xleaf",
                compact_model_param_name("gm_eq", "xleaf"),
            ),
        ));
        let catalog = MacroCatalog::from_macros([leaf, parent]).unwrap();

        assert!(validate_macro_catalog(&catalog, &primitive_catalog()).is_empty());
    }

    #[test]
    fn validates_compact_output_and_interface_bindings() {
        let unbound = Macro::new(
            "unbound",
            ports(),
            Circuit::builder()
                .primitive(
                    "xcore",
                    "stage_primitive",
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .build(),
            compact_model(),
        );
        let errors = validate_macro_catalog(
            &MacroCatalog::from_macros([unbound]).unwrap(),
            &primitive_catalog(),
        );
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::MissingCompactOutputBinding { parameter, .. }
                if parameter == "gm_eq"
        )));

        let invalid = leaf_macro()
            .with_compact_output(MacroCompactOutputBinding::new(
                "gm_eq",
                MacroOutputSource::candidate_column("missing", ""),
            ))
            .with_compact_output(MacroCompactOutputBinding::new(
                "unknown",
                MacroOutputSource::candidate_column("xcore", "xcore.value"),
            ))
            .with_interface_binding(MacroInterfaceBinding::new(
                "UNKNOWN_PORT",
                MacroOutputSource::ac_metric("missing", crate::analysis::AcMetric::DcGainDb),
            ));
        let errors = validate_macro_catalog(
            &MacroCatalog::from_macros([invalid]).unwrap(),
            &primitive_catalog(),
        );
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::DuplicateCompactOutputParameter { parameter, .. }
                if parameter == "gm_eq"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::UnknownCompactOutputParameter { parameter, .. }
                if parameter == "unknown"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::UnknownInterfacePort { port, .. }
                if port == "UNKNOWN_PORT"
        )));
        assert!(
            errors
                .iter()
                .filter(|error| matches!(error, MacroValidationError::InvalidOutputSource { .. }))
                .count()
                >= 2
        );
    }

    #[test]
    fn accumulates_connection_and_reference_errors() {
        let invalid = Macro::new(
            "invalid",
            ports(),
            Circuit::builder()
                .primitive(
                    "xcore",
                    "stage_primitive",
                    [("VIN", "VIN"), ("VIN", "VOUT"), ("EXTRA", "")],
                )
                .primitive("xmissing", "unknown", [("p", "VSS")])
                .build(),
            compact_model(),
        );
        let catalog = MacroCatalog::from_macros([invalid]).unwrap();
        let errors = validate_macro_catalog(&catalog, &primitive_catalog());

        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::DuplicateConnection { port, .. } if port == "VIN"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::MissingBlockConnection { port, .. } if port == "VSS"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::UnknownBlockPort { port, .. } if port == "EXTRA"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::UnknownPrimitive { primitive, .. } if primitive == "unknown"
        )));
    }

    #[test]
    fn rejects_non_linear_and_invalid_compact_model_entries() {
        let compact = Circuit::builder()
            .primitive(
                "xcore",
                "stage_primitive",
                [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
            )
            .resistor("bad_parameter", "VOUT", "VSS", " ")
            .capacitor("bad_constant", "VOUT", "VSS", f64::NAN)
            .build();
        let invalid = Macro::new(
            "invalid_compact",
            ports(),
            leaf_macro().circuit().clone(),
            compact,
        );
        let catalog = MacroCatalog::from_macros([invalid]).unwrap();
        let errors = validate_macro_catalog(&catalog, &primitive_catalog());

        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::NonLinearCompactBlock { block, .. }
                if block == "stage_primitive"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::EmptyParameter {
                circuit: MacroCircuitKind::CompactModel,
                ..
            }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::NonFiniteConstant {
                circuit: MacroCircuitKind::CompactModel,
                ..
            }
        )));
    }

    #[test]
    fn validates_primitive_small_signal_branches_against_manifest_pins() {
        let mut primitives = primitive_catalog();
        let mut primitive = primitives.get("stage_primitive").unwrap().clone();
        primitive.small_signal = Some(SmallSignalModel::new(vec![
            SmallSignalBranch::new("m1", "UNKNOWN", "", "VSS", ""),
            SmallSignalBranch::new("m1", "VOUT", "VIN", "VSS", "VSS"),
        ]));
        primitives.register(primitive);
        let macros = MacroCatalog::from_macros([leaf_macro()]).unwrap();

        let errors = validate_macro_catalog(&macros, &primitives);

        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::UnknownSmallSignalBranchPin {
                terminal: "drain",
                pin,
                ..
            } if pin == "UNKNOWN"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::EmptySmallSignalBranchPin {
                terminal: "gate",
                ..
            }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::EmptySmallSignalBranchPin {
                terminal: "bulk",
                ..
            }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::DuplicateSmallSignalBranch { branch, .. }
                if branch == "m1"
        )));
    }

    #[test]
    fn validates_physical_branch_ports_and_primitive_pins() {
        let mut primitives = primitive_catalog();
        let mut primitive = primitives.get("stage_primitive").unwrap().clone();
        primitive.physical_model = Some(PrimitivePhysicalModel::new(
            "",
            "missing",
            [
                ("", ""),
                ("D", "UNKNOWN"),
                ("D", "VIN"),
            ],
        ));
        primitives.register(primitive);
        let macros = MacroCatalog::from_macros([leaf_macro()]).unwrap();

        let errors = validate_macro_catalog(&macros, &primitives);

        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::EmptyPhysicalLutPrimitive { .. }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::UnknownPhysicalOperatingPointBranch { branch, .. }
                if branch == "missing"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::EmptyPhysicalPort { .. }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::DuplicatePhysicalPort { port, .. } if port == "D"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::EmptyPhysicalPin { .. }
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::UnknownPhysicalPin { pin, .. } if pin == "UNKNOWN"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::MissingPhysicalPinMapping { pin, .. }
                if pin == "VOUT" || pin == "VSS"
        )));
    }

    #[test]
    fn validates_testbench_identity_source_transfer_and_ac_settings() {
        let invalid_config = AdaptiveAcConfig {
            min_frequency_hz: 10.0,
            max_frequency_hz: 1.0,
            coarse_points_per_decade: 0,
            crossing_relative_tolerance: 0.0,
            max_refinement_steps: 0,
            retain_samples: false,
        };
        let analysis = AcAnalysis::new(
            TransferFunction::new("", ""),
            invalid_config,
            AdaptiveAcPolicy {
                mode: AnalysisMode::Prune,
                targets: AnalysisTargets::NONE,
                metrics: AcMetricSet::EMPTY,
            },
        );
        let invalid = leaf_macro()
            .with_ac_testbench(MacroAcTestbench::from_spice("gain", " ", analysis.clone()))
            .with_ac_testbench(MacroAcTestbench::from_spice("gain", " ", analysis));
        let catalog = MacroCatalog::from_macros([invalid]).unwrap();
        let errors = validate_macro_catalog(&catalog, &primitive_catalog());

        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::DuplicateTestbench { testbench, .. } if testbench == "gain"
        )));
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, MacroValidationError::EmptyTestbenchSource { .. }))
        );
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::EmptyTransferNode { node: "input", .. }
        )));
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, MacroValidationError::InvalidAcConfig { .. }))
        );
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, MacroValidationError::InvalidAcPolicy { .. }))
        );
    }

    #[test]
    fn detects_macro_dependency_cycles() {
        fn cyclic_macro(name: &str, dependency: &str) -> Macro {
            let circuit = Circuit::builder()
                .macro_instance("xnext", dependency, [("P", "P")])
                .build();
            Macro::new(
                name,
                vec![MacroPort::new("P", MacroPortRole::Inout)],
                circuit,
                Circuit::builder().resistor("r", "P", "P", 1.0).build(),
            )
        }

        let catalog = MacroCatalog::from_macros([
            cyclic_macro("stage_a", "stage_b"),
            cyclic_macro("stage_b", "stage_a"),
        ])
        .unwrap();
        let errors = validate_macro_catalog(&catalog, &PrimitiveCatalog::new());

        assert!(errors.iter().any(|error| matches!(
            error,
            MacroValidationError::CyclicDependency { path }
                if path == &["stage_a", "stage_b", "stage_a"]
        )));
    }
}
