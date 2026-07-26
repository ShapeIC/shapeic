use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use symbolica::prelude::{Atom, AtomCore, EvaluationError, ExpressionEvaluator};

/// A compiled real-valued Symbolica expression that reuses its evaluation stack.
#[derive(Clone, Debug)]
pub struct PreparedRealEvaluator {
    parameter_names: Vec<String>,
    evaluator: ExpressionEvaluator<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreparedEvaluationError {
    ParameterMismatch {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    InvalidParameterCount {
        expected: usize,
        actual: usize,
    },
    NonRealCoefficients,
    Symbolica(EvaluationError),
}

impl PreparedRealEvaluator {
    /// Compiles `expression` using the exact input order in `parameter_order`.
    ///
    /// The requested names must match all free symbols in the expression exactly.
    pub fn new(
        expression: &Atom,
        parameter_order: &[&str],
    ) -> Result<Self, PreparedEvaluationError> {
        let mut symbols_by_name = expression
            .get_all_symbols(false)
            .into_iter()
            .map(|symbol| (symbol.to_string(), Atom::from(symbol)))
            .collect::<BTreeMap<_, _>>();
        let expected = symbols_by_name.keys().cloned().collect::<Vec<_>>();
        let actual = parameter_order
            .iter()
            .map(|parameter| (*parameter).to_owned())
            .collect::<Vec<_>>();
        let actual_unique = actual.iter().cloned().collect::<BTreeSet<_>>();
        let expected_set = expected.iter().cloned().collect::<BTreeSet<_>>();
        if actual.len() != actual_unique.len() || actual_unique != expected_set {
            return Err(PreparedEvaluationError::ParameterMismatch { expected, actual });
        }

        let symbols = parameter_order
            .iter()
            .map(|parameter| {
                symbols_by_name
                    .remove(*parameter)
                    .expect("validated parameter must have a matching symbol")
            })
            .collect::<Vec<_>>();
        let evaluator = expression.evaluator(&symbols).build()?;
        if !evaluator.is_real() {
            return Err(PreparedEvaluationError::NonRealCoefficients);
        }
        let evaluator = evaluator.map_coeff(&|coefficient| coefficient.re.to_f64());

        Ok(Self {
            parameter_names: actual,
            evaluator,
        })
    }

    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Evaluates the prepared expression without rebuilding its instruction graph.
    ///
    /// The mutable receiver lets Symbolica reuse its internal scratch stack.
    pub fn evaluate(&mut self, values: &[f64]) -> Result<f64, PreparedEvaluationError> {
        if values.len() != self.parameter_names.len() {
            return Err(PreparedEvaluationError::InvalidParameterCount {
                expected: self.parameter_names.len(),
                actual: values.len(),
            });
        }

        let mut result = [0.0];
        self.evaluator.try_evaluate(values, &mut result)?;
        Ok(result[0])
    }
}

impl fmt::Display for PreparedEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParameterMismatch { expected, actual } => write!(
                formatter,
                "expression parameters {expected:?} do not match requested order {actual:?}"
            ),
            Self::InvalidParameterCount { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} parameter values, received {actual}"
                )
            }
            Self::NonRealCoefficients => {
                formatter.write_str("expression contains non-real coefficients")
            }
            Self::Symbolica(error) => write!(formatter, "could not evaluate expression: {error}"),
        }
    }
}

impl Error for PreparedEvaluationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Symbolica(error) => Some(error),
            _ => None,
        }
    }
}

impl From<EvaluationError> for PreparedEvaluationError {
    fn from(error: EvaluationError) -> Self {
        Self::Symbolica(error)
    }
}

#[cfg(test)]
mod tests {
    use super::{PreparedEvaluationError, PreparedRealEvaluator};
    use symbolica::prelude::parse;

    #[test]
    fn evaluates_parameters_in_the_requested_order() {
        let expression = parse!("a*b+c");
        let mut evaluator =
            PreparedRealEvaluator::new(&expression, &["c", "a", "b"]).expect("should prepare");

        assert_eq!(evaluator.parameter_names(), ["c", "a", "b"]);
        assert_eq!(evaluator.evaluate(&[1.0, 2.0, 3.0]).unwrap(), 7.0);
    }

    #[test]
    fn reuses_the_evaluator_with_new_values() {
        let expression = parse!("a*b+c");
        let mut evaluator =
            PreparedRealEvaluator::new(&expression, &["c", "a", "b"]).expect("should prepare");

        assert_eq!(evaluator.evaluate(&[1.0, 2.0, 3.0]).unwrap(), 7.0);
        assert_eq!(evaluator.evaluate(&[4.0, 5.0, 6.0]).unwrap(), 34.0);
    }

    #[test]
    fn rejects_missing_extra_and_duplicate_parameters() {
        let expression = parse!("a*b+c");

        for parameters in [
            &["a", "b"][..],
            &["a", "b", "c", "d"][..],
            &["a", "b", "b"][..],
        ] {
            assert!(matches!(
                PreparedRealEvaluator::new(&expression, parameters),
                Err(PreparedEvaluationError::ParameterMismatch { .. })
            ));
        }
    }

    #[test]
    fn rejects_an_incorrect_value_count() {
        let expression = parse!("a*b+c");
        let mut evaluator =
            PreparedRealEvaluator::new(&expression, &["a", "b", "c"]).expect("should prepare");

        assert_eq!(
            evaluator.evaluate(&[1.0, 2.0]),
            Err(PreparedEvaluationError::InvalidParameterCount {
                expected: 3,
                actual: 2,
            })
        );
    }
}
