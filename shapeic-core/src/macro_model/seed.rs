//! Validation and instance scoping of nominal compact seeds.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use crate::circuit::{CircuitValue, LinearElement};
use crate::exploration::candidate::{CandidatePoint, CandidateSet, candidate_column_name};
use crate::exploration::filter::CandidateFilter;
use crate::netlist::names::compact_model_param_name;

use super::{CompactMacroInstanceExplorationInput, Macro, MacroCompactSeed};

/// One invalid compact seed declaration or projection request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroCompactSeedError {
    MissingSeed { macro_name: String },
    EmptyInstancePath,
    EmptyParameterName,
    DuplicateParameter { parameter: String },
    MissingParameter { parameter: String },
    UnknownParameter { parameter: String },
    NonFiniteParameter { parameter: String },
    EmptyInterfacePort,
    DuplicateInterfacePort { port: String },
    MissingInterfacePort { port: String },
    UnknownInterfacePort { port: String },
    NonFiniteInterfaceValue { port: String },
}

impl fmt::Display for MacroCompactSeedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSeed { macro_name } => {
                write!(
                    formatter,
                    "macro '{macro_name}' has no nominal compact seed"
                )
            }
            Self::EmptyInstancePath => formatter.write_str("compact seed instance path is empty"),
            Self::EmptyParameterName => formatter.write_str("compact seed parameter name is empty"),
            Self::DuplicateParameter { parameter } => {
                write!(
                    formatter,
                    "compact seed parameter '{parameter}' is duplicated"
                )
            }
            Self::MissingParameter { parameter } => {
                write!(
                    formatter,
                    "compact seed is missing required parameter '{parameter}'"
                )
            }
            Self::UnknownParameter { parameter } => {
                write!(
                    formatter,
                    "compact seed contains unknown parameter '{parameter}'"
                )
            }
            Self::NonFiniteParameter { parameter } => {
                write!(
                    formatter,
                    "compact seed parameter '{parameter}' is not finite"
                )
            }
            Self::EmptyInterfacePort => formatter.write_str("compact seed interface port is empty"),
            Self::DuplicateInterfacePort { port } => {
                write!(
                    formatter,
                    "compact seed interface port '{port}' is duplicated"
                )
            }
            Self::MissingInterfacePort { port } => {
                write!(formatter, "compact seed is missing interface port '{port}'")
            }
            Self::UnknownInterfacePort { port } => {
                write!(
                    formatter,
                    "compact seed contains unknown interface port '{port}'"
                )
            }
            Self::NonFiniteInterfaceValue { port } => {
                write!(
                    formatter,
                    "compact seed interface value '{port}' is not finite"
                )
            }
        }
    }
}

impl Error for MacroCompactSeedError {}

pub(super) fn validate_compact_seed(
    macro_: &Macro,
    seed: &MacroCompactSeed,
) -> Vec<MacroCompactSeedError> {
    let required_parameters = compact_parameters(macro_);
    let required_interfaces = macro_
        .exploration()
        .interface_bindings()
        .iter()
        .map(|binding| binding.port())
        .collect::<BTreeSet<_>>();
    let mut errors = Vec::new();
    let mut parameters = BTreeSet::new();
    for (parameter, value) in seed.compact_parameters() {
        if parameter.trim().is_empty() {
            errors.push(MacroCompactSeedError::EmptyParameterName);
            continue;
        }
        if !parameters.insert(parameter.as_str()) {
            errors.push(MacroCompactSeedError::DuplicateParameter {
                parameter: parameter.clone(),
            });
        }
        if !required_parameters.contains(parameter.as_str()) {
            errors.push(MacroCompactSeedError::UnknownParameter {
                parameter: parameter.clone(),
            });
        }
        if !value.is_finite() {
            errors.push(MacroCompactSeedError::NonFiniteParameter {
                parameter: parameter.clone(),
            });
        }
    }
    for parameter in required_parameters.difference(&parameters) {
        errors.push(MacroCompactSeedError::MissingParameter {
            parameter: (*parameter).to_owned(),
        });
    }

    let mut interfaces = BTreeSet::new();
    for (port, value) in seed.interface_values() {
        if port.trim().is_empty() {
            errors.push(MacroCompactSeedError::EmptyInterfacePort);
            continue;
        }
        if !interfaces.insert(port.as_str()) {
            errors.push(MacroCompactSeedError::DuplicateInterfacePort { port: port.clone() });
        }
        if !required_interfaces.contains(port.as_str()) {
            errors.push(MacroCompactSeedError::UnknownInterfacePort { port: port.clone() });
        }
        if !value.is_finite() {
            errors.push(MacroCompactSeedError::NonFiniteInterfaceValue { port: port.clone() });
        }
    }
    for port in required_interfaces.difference(&interfaces) {
        errors.push(MacroCompactSeedError::MissingInterfacePort {
            port: (*port).to_owned(),
        });
    }
    errors
}

/// Validates that a macro has one complete nominal compact seed.
pub fn validate_macro_compact_seed(macro_: &Macro) -> Vec<MacroCompactSeedError> {
    match macro_.exploration().compact_seed() {
        Some(seed) => validate_compact_seed(macro_, seed),
        None => vec![MacroCompactSeedError::MissingSeed {
            macro_name: macro_.name().to_owned(),
        }],
    }
}

pub(super) fn build_compact_seed_input(
    macro_: &Macro,
    instance_path: String,
    filters: Vec<CandidateFilter>,
) -> Result<CompactMacroInstanceExplorationInput, Vec<MacroCompactSeedError>> {
    if instance_path.trim().is_empty() {
        return Err(vec![MacroCompactSeedError::EmptyInstancePath]);
    }
    let errors = validate_macro_compact_seed(macro_);
    if !errors.is_empty() {
        return Err(errors);
    }
    let seed = macro_
        .exploration()
        .compact_seed()
        .expect("successful seed validation resolved the compact seed");

    let mut values =
        Vec::with_capacity(seed.compact_parameters().len() + seed.interface_values().len());
    values.extend(
        seed.compact_parameters().iter().map(|(parameter, value)| {
            (compact_model_param_name(parameter, &instance_path), *value)
        }),
    );
    values.extend(seed.interface_values().iter().map(|(port, value)| {
        (
            candidate_column_name(&instance_path, &port.to_ascii_lowercase()),
            *value,
        )
    }));
    Ok(CompactMacroInstanceExplorationInput {
        candidates: CandidateSet::new(instance_path.clone(), vec![CandidatePoint::new(values)]),
        filters,
        interface_ports: seed
            .interface_values()
            .iter()
            .map(|(port, _)| port.clone())
            .collect(),
        provenance: None,
    })
}

fn compact_parameters(macro_: &Macro) -> BTreeSet<&str> {
    macro_
        .compact_model()
        .instances()
        .iter()
        .filter_map(|instance| match instance.block() {
            crate::circuit::BlockRef::Element(element) => Some(element_value(element)),
            _ => None,
        })
        .filter_map(|value| match value {
            CircuitValue::Parameter(parameter) => Some(parameter.as_str()),
            CircuitValue::Constant(_) => None,
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use crate::circuit::Circuit;
    use crate::macro_model::{MacroInterfaceBinding, MacroOutputSource};

    use super::*;

    fn child(seed: MacroCompactSeed) -> Macro {
        Macro::new(
            "child",
            Vec::new(),
            Circuit::default(),
            Circuit::builder()
                .resistor("r", "OUT", "0", "r_eq")
                .vccs("g", "OUT", "0", "IN", "0", "gm_eq")
                .build(),
        )
        .with_interface_binding(MacroInterfaceBinding::new(
            "OUT",
            MacroOutputSource::candidate_column("x", "x.out"),
        ))
        .with_compact_seed(seed)
    }

    #[test]
    fn scopes_one_valid_seed_for_each_parent_instance() {
        let macro_ = child(MacroCompactSeed::new(
            [("r_eq", 10.0), ("gm_eq", 2.0)],
            [("OUT", 0.7)],
        ));

        let first = build_compact_seed_input(&macro_, "x1".to_owned(), Vec::new()).unwrap();
        let second = build_compact_seed_input(&macro_, "x2".to_owned(), Vec::new()).unwrap();

        assert_eq!(first.candidates.points[0].get("r_eq__x1"), Some(10.0));
        assert_eq!(first.candidates.points[0].get("x1.out"), Some(0.7));
        assert_eq!(first.interface_ports, ["OUT"]);
        assert_eq!(second.candidates.points[0].get("r_eq__x2"), Some(10.0));
        assert!(first.provenance.is_none());
    }

    #[test]
    fn rejects_missing_unknown_duplicate_and_non_finite_seed_fields() {
        let macro_ = child(MacroCompactSeed::new(
            [("r_eq", 1.0), ("r_eq", 2.0), ("extra", f64::NAN)],
            [("OTHER", 0.0)],
        ));
        let errors = validate_compact_seed(&macro_, macro_.exploration().compact_seed().unwrap());

        assert!(errors.iter().any(|error| matches!(
            error,
            MacroCompactSeedError::DuplicateParameter { parameter } if parameter == "r_eq"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroCompactSeedError::MissingParameter { parameter } if parameter == "gm_eq"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroCompactSeedError::UnknownInterfacePort { port } if port == "OTHER"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroCompactSeedError::MissingInterfacePort { port } if port == "OUT"
        )));
    }

    #[test]
    fn reports_a_missing_seed_for_a_potential_child() {
        let macro_ = Macro::new(
            "child",
            Vec::new(),
            Circuit::default(),
            Circuit::builder().resistor("r", "OUT", "0", "r_eq").build(),
        );

        assert_eq!(
            validate_macro_compact_seed(&macro_),
            [MacroCompactSeedError::MissingSeed {
                macro_name: "child".to_owned(),
            }]
        );
    }
}
