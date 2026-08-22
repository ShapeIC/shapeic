use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::{BlockRef, Circuit, CircuitValue, LinearElement};

use super::{Macro, MacroCatalog, MacroTestbenchSource};

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
                Some(manifest) => Some((
                    primitive.clone(),
                    manifest
                        .pins
                        .iter()
                        .map(|pin| pin.name.as_str())
                        .collect::<Vec<_>>(),
                )),
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
    use crate::macro_model::{Macro, MacroAcTestbench, MacroCatalog, MacroPort, MacroPortRole};
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};
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
        );
        let catalog = MacroCatalog::from_macros([leaf, parent]).unwrap();

        assert!(validate_macro_catalog(&catalog, &primitive_catalog()).is_empty());
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
