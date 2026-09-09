//! Shape-preserving reductions used by parent-to-child derivation rules.

use std::error::Error;
use std::fmt;

use super::{MacroDerivationReduction, MacroDerivedValue};

/// An invalid source set for one derivation reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroDerivationReductionError {
    EmptySource,
    NonFiniteSource { index: usize },
}

impl fmt::Display for MacroDerivationReductionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySource => formatter.write_str("derivation reduction has no source rows"),
            Self::NonFiniteSource { index } => {
                write!(formatter, "derivation source row {index} is not finite")
            }
        }
    }
}

impl Error for MacroDerivationReductionError {}

pub(super) fn reduce_derivation_values(
    reduction: MacroDerivationReduction,
    values: impl IntoIterator<Item = f64>,
) -> Result<MacroDerivedValue, MacroDerivationReductionError> {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() {
        return Err(MacroDerivationReductionError::EmptySource);
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MacroDerivationReductionError::NonFiniteSource { index });
    }
    Ok(match reduction {
        MacroDerivationReduction::Minimum => {
            MacroDerivedValue::Scalar(values.iter().copied().reduce(f64::min).expect("non-empty"))
        }
        MacroDerivationReduction::Maximum => {
            MacroDerivedValue::Scalar(values.iter().copied().reduce(f64::max).expect("non-empty"))
        }
        MacroDerivationReduction::Range => MacroDerivedValue::Range {
            minimum: values.iter().copied().reduce(f64::min).expect("non-empty"),
            maximum: values.iter().copied().reduce(f64::max).expect("non-empty"),
        },
        MacroDerivationReduction::UniqueValues => {
            let mut unique = Vec::new();
            for value in values {
                if !unique.contains(&value) {
                    unique.push(value);
                }
            }
            MacroDerivedValue::UniqueValues(unique)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduces_scalars_ranges_and_stable_unique_values() {
        let values = [3.0, 1.0, 3.0, 2.0];
        assert_eq!(
            reduce_derivation_values(MacroDerivationReduction::Minimum, values).unwrap(),
            MacroDerivedValue::Scalar(1.0)
        );
        assert_eq!(
            reduce_derivation_values(MacroDerivationReduction::Maximum, values).unwrap(),
            MacroDerivedValue::Scalar(3.0)
        );
        assert_eq!(
            reduce_derivation_values(MacroDerivationReduction::Range, values).unwrap(),
            MacroDerivedValue::Range {
                minimum: 1.0,
                maximum: 3.0,
            }
        );
        assert_eq!(
            reduce_derivation_values(MacroDerivationReduction::UniqueValues, values).unwrap(),
            MacroDerivedValue::UniqueValues(vec![3.0, 1.0, 2.0])
        );
    }

    #[test]
    fn rejects_empty_and_non_finite_sources() {
        assert_eq!(
            reduce_derivation_values(MacroDerivationReduction::Minimum, []).unwrap_err(),
            MacroDerivationReductionError::EmptySource
        );
        assert_eq!(
            reduce_derivation_values(MacroDerivationReduction::Range, [1.0, f64::NAN]).unwrap_err(),
            MacroDerivationReductionError::NonFiniteSource { index: 1 }
        );
    }
}
