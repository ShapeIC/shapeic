//! Validation, transposition, and instance scoping of aligned compact seeds.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use crate::circuit::{CircuitValue, LinearElement};
use crate::exploration::candidate::{CandidatePoint, CandidateSet};
use crate::exploration::filter::CandidateFilter;
use crate::netlist::names::compact_model_param_name;

use super::{CompactMacroInstanceExplorationInput, Macro, MacroCompactSeedSet};

/// One invalid compact seed declaration or projection request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroCompactSeedError {
    MissingSeeds {
        macro_name: String,
    },
    EmptySeedSet,
    EmptyInstancePath,
    EmptyParameterName,
    DuplicateParameter {
        parameter: String,
    },
    MissingParameter {
        parameter: String,
    },
    UnknownParameter {
        parameter: String,
    },
    EmptyParameterValues {
        parameter: String,
    },
    ParameterLengthMismatch {
        parameter: String,
        expected: usize,
        actual: usize,
    },
    NonFiniteParameter {
        parameter: String,
        index: usize,
    },
}

impl fmt::Display for MacroCompactSeedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSeeds { macro_name } => {
                write!(formatter, "macro '{macro_name}' has no compact seed set")
            }
            Self::EmptySeedSet => formatter.write_str("compact seed set contains no rows"),
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
            Self::EmptyParameterValues { parameter } => {
                write!(
                    formatter,
                    "compact seed parameter '{parameter}' has no values"
                )
            }
            Self::ParameterLengthMismatch {
                parameter,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "compact seed parameter '{parameter}' has {actual} values; expected {expected}"
                )
            }
            Self::NonFiniteParameter { parameter, index } => {
                write!(
                    formatter,
                    "compact seed parameter '{parameter}' is not finite at row {index}"
                )
            }
        }
    }
}

impl Error for MacroCompactSeedError {}

pub(super) fn validate_compact_seeds(
    macro_: &Macro,
    seeds: &MacroCompactSeedSet,
) -> Vec<MacroCompactSeedError> {
    let required_parameters = compact_parameters(macro_);
    let mut errors = Vec::new();
    if seeds.is_empty() {
        errors.push(MacroCompactSeedError::EmptySeedSet);
    }
    let mut parameters = BTreeSet::new();
    for (parameter, values) in seeds.compact_parameters() {
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
        if values.is_empty() {
            errors.push(MacroCompactSeedError::EmptyParameterValues {
                parameter: parameter.clone(),
            });
        } else if values.len() != seeds.len() {
            errors.push(MacroCompactSeedError::ParameterLengthMismatch {
                parameter: parameter.clone(),
                expected: seeds.len(),
                actual: values.len(),
            });
        }
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                errors.push(MacroCompactSeedError::NonFiniteParameter {
                    parameter: parameter.clone(),
                    index,
                });
            }
        }
    }
    for parameter in required_parameters.difference(&parameters) {
        errors.push(MacroCompactSeedError::MissingParameter {
            parameter: (*parameter).to_owned(),
        });
    }
    errors
}

/// Validates that a macro has a complete aligned compact seed set.
pub fn validate_macro_compact_seeds(macro_: &Macro) -> Vec<MacroCompactSeedError> {
    match macro_.exploration().compact_seeds() {
        Some(seeds) => validate_compact_seeds(macro_, seeds),
        None => vec![MacroCompactSeedError::MissingSeeds {
            macro_name: macro_.name().to_owned(),
        }],
    }
}

pub(super) fn build_compact_seed_set_input(
    macro_: &Macro,
    instance_path: String,
    filters: Vec<CandidateFilter>,
) -> Result<CompactMacroInstanceExplorationInput, Vec<MacroCompactSeedError>> {
    if instance_path.trim().is_empty() {
        return Err(vec![MacroCompactSeedError::EmptyInstancePath]);
    }
    let errors = validate_macro_compact_seeds(macro_);
    if !errors.is_empty() {
        return Err(errors);
    }
    let seeds = macro_
        .exploration()
        .compact_seeds()
        .expect("successful seed validation resolved the compact seed set");
    let points = (0..seeds.len())
        .map(|index| {
            CandidatePoint::new(
                seeds
                    .compact_parameters()
                    .iter()
                    .map(|(parameter, values)| {
                        (
                            compact_model_param_name(parameter, &instance_path),
                            values[index],
                        )
                    })
                    .collect(),
            )
        })
        .collect();
    Ok(CompactMacroInstanceExplorationInput {
        candidates: CandidateSet::new(instance_path, points),
        filters,
        interface_ports: Vec::new(),
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
    use super::*;
    use crate::circuit::Circuit;

    fn child(seeds: MacroCompactSeedSet) -> Macro {
        Macro::new(
            "child",
            Vec::new(),
            Circuit::default(),
            Circuit::builder()
                .resistor("r", "OUT", "0", "r_eq")
                .vccs("g", "OUT", "0", "IN", "0", "gm_eq")
                .build(),
        )
        .with_compact_seeds(seeds)
    }

    #[test]
    fn transposes_and_scopes_aligned_seeds_for_each_parent_instance() {
        let macro_ = child(MacroCompactSeedSet::aligned([
            ("r_eq", vec![10.0, 20.0]),
            ("gm_eq", vec![2.0, 3.0]),
        ]));

        let first = build_compact_seed_set_input(&macro_, "x1".to_owned(), Vec::new()).unwrap();
        let second = build_compact_seed_set_input(&macro_, "x2".to_owned(), Vec::new()).unwrap();

        assert_eq!(first.candidates.points[0].get("r_eq__x1"), Some(10.0));
        assert_eq!(first.candidates.points[0].get("gm_eq__x1"), Some(2.0));
        assert_eq!(first.candidates.points[1].get("r_eq__x1"), Some(20.0));
        assert_eq!(first.candidates.points[1].get("gm_eq__x1"), Some(3.0));
        assert!(first.interface_ports.is_empty());
        assert_eq!(second.candidates.points[0].get("r_eq__x2"), Some(10.0));
        assert!(first.provenance.is_none());
    }

    #[test]
    fn rejects_invalid_parameter_names_values_and_lengths() {
        let macro_ = child(MacroCompactSeedSet::aligned([
            ("r_eq", vec![1.0, 2.0]),
            ("r_eq", vec![3.0, 4.0]),
            ("extra", vec![f64::NAN]),
        ]));
        let errors = validate_compact_seeds(&macro_, macro_.exploration().compact_seeds().unwrap());

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
            MacroCompactSeedError::UnknownParameter { parameter } if parameter == "extra"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroCompactSeedError::ParameterLengthMismatch { parameter, expected: 2, actual: 1 }
                if parameter == "extra"
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            MacroCompactSeedError::NonFiniteParameter { parameter, index: 0 }
                if parameter == "extra"
        )));
    }

    #[test]
    fn supports_one_seed_row_for_a_parameter_free_compact_model() {
        let macro_ = Macro::new(
            "constant",
            Vec::new(),
            Circuit::default(),
            Circuit::builder().resistor("r", "OUT", "0", 10.0).build(),
        )
        .with_compact_seeds(MacroCompactSeedSet::constant());

        let input = build_compact_seed_set_input(&macro_, "x1".to_owned(), Vec::new()).unwrap();
        assert_eq!(input.candidates.points.len(), 1);
        assert!(input.candidates.points[0].values.is_empty());
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
            validate_macro_compact_seeds(&macro_),
            [MacroCompactSeedError::MissingSeeds {
                macro_name: "child".to_owned(),
            }]
        );
    }
}
