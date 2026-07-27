use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use nalgebra::{DMatrix, DVector};
use ndarray::ArrayView2;
use num_complex::Complex64;
use symbolica::prelude::{Atom, AtomCore, EvaluationError, ExpressionEvaluator};

use crate::mna::MnaResult;
use crate::spice_parser::NodeMap;

/// A compiled, frequency-independent MNA template.
///
/// The symbolic base system is compiled once. Each parameter set can then be
/// instantiated as a numerical DC matrix before numerical capacitances are
/// stamped into the resulting [`NumericMnaSystem`].
#[derive(Clone, Debug)]
pub struct PreparedNumericMna {
    parameter_names: Vec<String>,
    evaluator: ExpressionEvaluator<f64>,
    outputs: Vec<f64>,
    rows: usize,
    matrix_len: usize,
    nodes: NodeMap,
}

/// A numerical linear MNA represented as `A(s) = G + sC`.
#[derive(Clone, Debug)]
pub struct NumericMnaSystem {
    dc: DMatrix<f64>,
    capacitance: DMatrix<f64>,
    rhs: DVector<f64>,
    nodes: NodeMap,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NumericMnaError {
    InvalidBaseDimensions {
        matrix_rows: usize,
        matrix_columns: usize,
        rhs_rows: usize,
        rhs_columns: usize,
    },
    FrequencyDependentBase,
    ParameterMismatch {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    InvalidParameterCount {
        expected: usize,
        actual: usize,
    },
    NonFiniteParameter {
        index: usize,
        value: f64,
    },
    NonRealCoefficients,
    Evaluation(EvaluationError),
    NonFiniteBaseCoefficient,
    InvalidPortAdmittance {
        message: String,
    },
    MissingNode(String),
    InvalidFrequency(f64),
    NonFiniteSystemCoefficient {
        frequency_hz: f64,
    },
}

impl PreparedNumericMna {
    /// Compile a frequency-independent symbolic MNA using an exact parameter order.
    pub fn new(system: &MnaResult, parameter_order: &[&str]) -> Result<Self, NumericMnaError> {
        let rows = system.a.nrows();
        let columns = system.a.ncols();
        if rows == 0 || rows != columns || system.z.nrows() != rows || system.z.ncols() != 1 {
            return Err(NumericMnaError::InvalidBaseDimensions {
                matrix_rows: rows,
                matrix_columns: columns,
                rhs_rows: system.z.nrows(),
                rhs_columns: system.z.ncols(),
            });
        }

        let matrix_len = rows * columns;
        let mut expressions = Vec::with_capacity(matrix_len + rows);
        for row in 0..rows {
            let row = u32::try_from(row).expect("MNA row count must fit Symbolica matrix indices");
            for column in 0..columns {
                let column = u32::try_from(column)
                    .expect("MNA column count must fit Symbolica matrix indices");
                expressions.push(system.a[(row, column)].clone());
            }
        }
        for row in 0..rows {
            let row = u32::try_from(row).expect("MNA row count must fit Symbolica matrix indices");
            expressions.push(system.z[(row, 0)].clone());
        }

        let mut symbols_by_name = BTreeMap::new();
        for expression in &expressions {
            for symbol in expression.get_all_symbols(false) {
                let symbol = Atom::from(symbol);
                symbols_by_name.entry(symbol.to_string()).or_insert(symbol);
            }
        }
        if symbols_by_name.contains_key("s") {
            return Err(NumericMnaError::FrequencyDependentBase);
        }

        let expected = symbols_by_name.keys().cloned().collect::<Vec<_>>();
        let actual = parameter_order
            .iter()
            .map(|parameter| (*parameter).to_owned())
            .collect::<Vec<_>>();
        let expected_set = expected.iter().cloned().collect::<BTreeSet<_>>();
        let actual_set = actual.iter().cloned().collect::<BTreeSet<_>>();
        if actual.len() != actual_set.len() || actual_set != expected_set {
            return Err(NumericMnaError::ParameterMismatch { expected, actual });
        }

        let symbols = parameter_order
            .iter()
            .map(|parameter| {
                symbols_by_name
                    .remove(*parameter)
                    .expect("validated parameter must have a matching symbol")
            })
            .collect::<Vec<_>>();
        let evaluator = Atom::evaluator_multiple(&expressions, &symbols)
            .build()
            .map_err(NumericMnaError::Evaluation)?;
        if !evaluator.is_real() {
            return Err(NumericMnaError::NonRealCoefficients);
        }
        let evaluator = evaluator.map_coeff(&|coefficient| coefficient.re.to_f64());

        Ok(Self {
            parameter_names: actual,
            evaluator,
            outputs: vec![0.0; expressions.len()],
            rows,
            matrix_len,
            nodes: system.nodes.clone(),
        })
    }

    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Evaluate the prepared base once and create a numerical `G + sC` system.
    pub fn instantiate(
        &mut self,
        parameter_values: &[f64],
    ) -> Result<NumericMnaSystem, NumericMnaError> {
        if parameter_values.len() != self.parameter_names.len() {
            return Err(NumericMnaError::InvalidParameterCount {
                expected: self.parameter_names.len(),
                actual: parameter_values.len(),
            });
        }
        if let Some((index, value)) = parameter_values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(NumericMnaError::NonFiniteParameter { index, value });
        }

        self.evaluator
            .try_evaluate(parameter_values, &mut self.outputs)
            .map_err(NumericMnaError::Evaluation)?;
        if self.outputs.iter().any(|value| !value.is_finite()) {
            return Err(NumericMnaError::NonFiniteBaseCoefficient);
        }

        Ok(NumericMnaSystem {
            dc: DMatrix::from_row_slice(self.rows, self.rows, &self.outputs[..self.matrix_len]),
            capacitance: DMatrix::zeros(self.rows, self.rows),
            rhs: DVector::from_row_slice(&self.outputs[self.matrix_len..]),
            nodes: self.nodes.clone(),
        })
    }
}

impl NumericMnaSystem {
    pub fn dc_matrix(&self) -> &DMatrix<f64> {
        &self.dc
    }

    pub fn capacitance_matrix(&self) -> &DMatrix<f64> {
        &self.capacitance
    }

    pub fn rhs(&self) -> &DVector<f64> {
        &self.rhs
    }

    /// Stamp a complete port model into the numerical `G` and `C` matrices.
    pub fn stamp_port_admittance(
        &mut self,
        port_order: &[String],
        connections: &BTreeMap<String, String>,
        conductance: ArrayView2<'_, f64>,
        capacitance: ArrayView2<'_, f64>,
    ) -> Result<(), NumericMnaError> {
        let port_count = port_order.len();
        let expected = [port_count, port_count];
        if conductance.shape() != expected || capacitance.shape() != expected {
            return Err(NumericMnaError::InvalidPortAdmittance {
                message: format!("G and C must both have shape {expected:?}"),
            });
        }
        if connections.len() != port_count {
            return Err(NumericMnaError::InvalidPortAdmittance {
                message: "each port must have exactly one circuit-node connection".to_owned(),
            });
        }

        let mut matrix_indices = Vec::with_capacity(port_count);
        for port in port_order {
            let node_name =
                connections
                    .get(port)
                    .ok_or_else(|| NumericMnaError::InvalidPortAdmittance {
                        message: format!("port '{port}' has no circuit-node connection"),
                    })?;
            let node_number = self
                .nodes
                .nodes
                .get(node_name)
                .copied()
                .ok_or_else(|| NumericMnaError::MissingNode(node_name.clone()))?;
            let matrix_index = node_number.checked_sub(1);
            if matrix_index.is_some_and(|index| index >= self.dc.nrows()) {
                return Err(NumericMnaError::InvalidPortAdmittance {
                    message: format!("node '{node_name}' is outside the MNA matrix"),
                });
            }
            matrix_indices.push(matrix_index);
        }

        for row in 0..port_count {
            let Some(mna_row) = matrix_indices[row] else {
                continue;
            };
            for column in 0..port_count {
                let Some(mna_column) = matrix_indices[column] else {
                    continue;
                };
                let g = conductance[(row, column)];
                let c = capacitance[(row, column)];
                if !g.is_finite() || !c.is_finite() {
                    return Err(NumericMnaError::InvalidPortAdmittance {
                        message: format!("non-finite coefficient at ({row}, {column})"),
                    });
                }
                self.dc[(mna_row, mna_column)] += g;
                self.capacitance[(mna_row, mna_column)] += c;
            }
        }
        Ok(())
    }

    /// Solve the numerical MNA at one frequency and return a node voltage.
    pub fn solve_node(
        &self,
        frequency_hz: f64,
        node_name: &str,
    ) -> Result<Option<Complex64>, NumericMnaError> {
        if !frequency_hz.is_finite() || frequency_hz < 0.0 {
            return Err(NumericMnaError::InvalidFrequency(frequency_hz));
        }
        let node_number = self
            .nodes
            .nodes
            .get(node_name)
            .copied()
            .ok_or_else(|| NumericMnaError::MissingNode(node_name.to_owned()))?;
        let output_row =
            node_number
                .checked_sub(1)
                .ok_or_else(|| NumericMnaError::InvalidPortAdmittance {
                    message: format!("cannot return ground node '{node_name}'"),
                })?;
        if output_row >= self.dc.nrows() {
            return Err(NumericMnaError::InvalidPortAdmittance {
                message: format!("node '{node_name}' is outside the MNA matrix"),
            });
        }

        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let matrix = DMatrix::from_fn(self.dc.nrows(), self.dc.ncols(), |row, column| {
            Complex64::new(
                self.dc[(row, column)],
                omega * self.capacitance[(row, column)],
            )
        });
        let rhs = self.rhs.map(|value| Complex64::new(value, 0.0));
        if matrix
            .iter()
            .chain(rhs.iter())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(NumericMnaError::NonFiniteSystemCoefficient { frequency_hz });
        }
        let Some(solution) = matrix.lu().solve(&rhs) else {
            return Ok(None);
        };
        let output = solution[output_row];
        if !output.re.is_finite() || !output.im.is_finite() {
            return Err(NumericMnaError::NonFiniteSystemCoefficient { frequency_hz });
        }
        Ok(Some(output))
    }
}

impl fmt::Display for NumericMnaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBaseDimensions {
                matrix_rows,
                matrix_columns,
                rhs_rows,
                rhs_columns,
            } => write!(
                formatter,
                "invalid base MNA dimensions: A={matrix_rows}x{matrix_columns}, \
                 z={rhs_rows}x{rhs_columns}"
            ),
            Self::FrequencyDependentBase => formatter.write_str(
                "prepared numerical MNA requires a frequency-independent base; stamp C separately",
            ),
            Self::ParameterMismatch { expected, actual } => write!(
                formatter,
                "MNA parameters {expected:?} do not match requested order {actual:?}"
            ),
            Self::InvalidParameterCount { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} parameter values, received {actual}"
                )
            }
            Self::NonFiniteParameter { index, value } => {
                write!(formatter, "parameter {index} is not finite: {value}")
            }
            Self::NonRealCoefficients => {
                formatter.write_str("base MNA contains non-real coefficients")
            }
            Self::Evaluation(error) => write!(formatter, "could not evaluate base MNA: {error}"),
            Self::NonFiniteBaseCoefficient => {
                formatter.write_str("base MNA evaluation produced a non-finite coefficient")
            }
            Self::InvalidPortAdmittance { message } => {
                write!(formatter, "invalid port admittance: {message}")
            }
            Self::MissingNode(node) => write!(formatter, "MNA node '{node}' is missing"),
            Self::InvalidFrequency(value) => {
                write!(formatter, "invalid AC frequency {value}")
            }
            Self::NonFiniteSystemCoefficient { frequency_hz } => write!(
                formatter,
                "numerical MNA contains a non-finite value at {frequency_hz:.6e} Hz"
            ),
        }
    }
}

impl Error for NumericMnaError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Evaluation(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use ndarray::array;
    use symbolica::domains::atom::AtomField;
    use symbolica::prelude::{Matrix, parse};

    use super::{NumericMnaError, PreparedNumericMna};
    use crate::mna::MnaResult;
    use crate::spice_parser::NodeMap;

    fn rc_low_pass() -> MnaResult {
        let mut a = Matrix::new(3, 3, AtomField::new());
        a[(0, 0)] = parse!("1/r");
        a[(0, 1)] = parse!("-1/r");
        a[(0, 2)] = parse!("1");
        a[(1, 0)] = parse!("-1/r");
        a[(1, 1)] = parse!("1/r");
        a[(2, 0)] = parse!("1");

        let mut z = Matrix::new(3, 1, AtomField::new());
        z[(2, 0)] = parse!("1");
        MnaResult {
            report: String::new(),
            a,
            x: Vec::new(),
            z,
            nodes: NodeMap {
                nodes: HashMap::from([
                    ("VIN".to_owned(), 1),
                    ("VOUT".to_owned(), 2),
                    ("VSS".to_owned(), 0),
                ]),
            },
        }
    }

    fn stamp_output_capacitance(
        system: &mut super::NumericMnaSystem,
        capacitance: f64,
    ) -> Result<(), NumericMnaError> {
        let ports = vec!["P".to_owned(), "N".to_owned()];
        let connections = BTreeMap::from([
            ("P".to_owned(), "VOUT".to_owned()),
            ("N".to_owned(), "VSS".to_owned()),
        ]);
        system.stamp_port_admittance(
            &ports,
            &connections,
            array![[0.0, 0.0], [0.0, 0.0]].view(),
            array![[capacitance, -capacitance], [-capacitance, capacitance]].view(),
        )
    }

    #[test]
    fn solves_a_parameterized_rc_low_pass() {
        let mut prepared = PreparedNumericMna::new(&rc_low_pass(), &["r"]).unwrap();
        let resistance = 1.0e3;
        let capacitance = 1.0e-9;
        let mut system = prepared.instantiate(&[resistance]).unwrap();
        stamp_output_capacitance(&mut system, capacitance).unwrap();

        let dc = system.solve_node(0.0, "VOUT").unwrap().unwrap();
        assert!((dc.re - 1.0).abs() < 1.0e-12);
        assert!(dc.im.abs() < 1.0e-12);

        let pole_hz = 1.0 / (2.0 * std::f64::consts::PI * resistance * capacitance);
        let at_pole = system.solve_node(pole_hz, "VOUT").unwrap().unwrap();
        assert!((at_pole.norm() - 2.0_f64.sqrt().recip()).abs() < 1.0e-12);
        assert!((at_pole.arg().to_degrees() + 45.0).abs() < 1.0e-10);
    }

    #[test]
    fn rejects_frequency_dependent_and_mismatched_base_systems() {
        let mut frequency_dependent = rc_low_pass();
        frequency_dependent.a[(1, 1)] += parse!("s");
        assert!(matches!(
            PreparedNumericMna::new(&frequency_dependent, &["r"]),
            Err(NumericMnaError::FrequencyDependentBase)
        ));
        assert!(matches!(
            PreparedNumericMna::new(&rc_low_pass(), &["wrong"]),
            Err(NumericMnaError::ParameterMismatch { .. })
        ));
    }

    #[test]
    fn validates_instantiation_and_port_stamps() {
        let mut prepared = PreparedNumericMna::new(&rc_low_pass(), &["r"]).unwrap();
        assert!(matches!(
            prepared.instantiate(&[]),
            Err(NumericMnaError::InvalidParameterCount { .. })
        ));
        assert!(matches!(
            prepared.instantiate(&[f64::NAN]),
            Err(NumericMnaError::NonFiniteParameter { .. })
        ));

        let mut system = prepared.instantiate(&[1.0e3]).unwrap();
        let ports = vec!["P".to_owned()];
        let connections = BTreeMap::from([("P".to_owned(), "VOUT".to_owned())]);
        assert!(matches!(
            system.stamp_port_admittance(
                &ports,
                &connections,
                array![[0.0]].view(),
                array![[f64::NAN]].view(),
            ),
            Err(NumericMnaError::InvalidPortAdmittance { .. })
        ));
    }

    #[test]
    fn collapses_ports_that_share_the_same_circuit_node() {
        let mut prepared = PreparedNumericMna::new(&rc_low_pass(), &["r"]).unwrap();
        let mut system = prepared.instantiate(&[1.0e3]).unwrap();
        let ports = vec!["A".to_owned(), "B".to_owned()];
        let connections = BTreeMap::from([
            ("A".to_owned(), "VOUT".to_owned()),
            ("B".to_owned(), "VOUT".to_owned()),
        ]);
        system
            .stamp_port_admittance(
                &ports,
                &connections,
                array![[0.0, 0.0], [0.0, 0.0]].view(),
                array![[2.0, -2.0], [-2.0, 2.0]].view(),
            )
            .unwrap();

        assert_eq!(system.capacitance_matrix()[(1, 1)], 0.0);
    }
}
