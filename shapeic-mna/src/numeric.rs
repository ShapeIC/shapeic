//! Numerical evaluation and AC solution of prepared MNA systems.
//!
//! This module converts a symbolic MNA system affine in the Laplace variable
//! into a reusable numerical evaluator. Symbolic circuit parameters are
//! compiled once with Symbolica and can then be evaluated repeatedly for
//! different parameter values.
//!
//! The instantiated numerical system is represented as
//!
//! ```text
//! A(s) = A0 + sC
//! ```
//!
//! where `A0` is the frequency-independent MNA matrix and `C` contains the
//! capacitance contributions. AC systems are evaluated using `s = jω` and
//! solved numerically with `nalgebra`.
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use nalgebra::{DMatrix, DVector};
use ndarray::ArrayView2;
use num_complex::Complex64;
use symbolica::prelude::{Atom, AtomCore, EvaluationError, ExpressionEvaluator};

use crate::mna::MnaResult;
use crate::spice2cir::NodeMap;

/// A compiled MNA template affine in the Laplace variable `s`.
///
/// Each symbolic matrix entry is separated into its constant and linear parts
/// so that `A(s) = A0 + sC`. Both matrices and the frequency-independent right-
/// hand side are compiled into one numerical evaluator. The resulting template
/// can then be instantiated repeatedly for different parameter values without
/// rebuilding the symbolic expressions.
///
/// Use [`PreparedNumericMna::instantiate`] to create a [`NumericMnaSystem`].
#[derive(Clone, Debug)]
pub struct PreparedNumericMna {
    parameter_names: Vec<String>,
    evaluator: ExpressionEvaluator<f64>,
    outputs: Vec<f64>,
    rows: usize,
    matrix_len: usize,
    nodes: NodeMap,
}

/// A numerical MNA system represented as `A(s) = A0 + sC`.
///
/// `A0` contains the frequency-independent part of the MNA system, while
/// the capacitance matrix contains the coefficients multiplied by `s`.
///
/// For AC analysis, the system matrix is evaluated as
///
/// ```text
/// A(jω) = A0 + jωC
/// ```
///
/// and solved numerically.
#[derive(Clone, Debug)]
pub struct NumericMnaSystem {
    base: DMatrix<f64>,
    capacitance: DMatrix<f64>,
    rhs: DVector<f64>,
    nodes: NodeMap,
}

/// Errors that may occur while preparing, instantiating, stamping, or solving
/// a numerical MNA system.
#[derive(Clone, Debug, PartialEq)]
pub enum NumericMnaError {
    /// The symbolic base MNA matrix or right-hand side has invalid dimensions.
    InvalidBaseDimensions {
        /// Number of rows in the MNA matrix.
        matrix_rows: usize,
        /// Number of columns in the MNA matrix.
        matrix_columns: usize,
        /// Number of rows in the right-hand-side matrix.
        rhs_rows: usize,
        /// Number of columns in the right-hand-side matrix.
        rhs_columns: usize,
    },
    /// An MNA matrix entry is not affine in the Laplace variable `s`.
    NonAffineFrequencyDependence {
        /// Zero-based matrix row containing the unsupported expression.
        row: usize,
        /// Zero-based matrix column containing the unsupported expression.
        column: usize,
        /// Unsupported symbolic matrix entry.
        expression: String,
    },
    /// The right-hand side contains the Laplace variable `s`.
    FrequencyDependentRhs {
        /// Zero-based right-hand-side row containing `s`.
        row: usize,
        /// Unsupported symbolic right-hand-side entry.
        expression: String,
    },
    /// The requested parameter order does not match the symbolic parameters
    /// present in the MNA system.
    ParameterMismatch {
        /// Parameters found in the symbolic MNA system.
        expected: Vec<String>,
        /// Parameter names provided by the caller.
        actual: Vec<String>,
    },
    /// The number of numerical parameter values does not match the prepared
    /// parameter list.
    InvalidParameterCount {
        /// Number of parameter values required by the prepared system.
        expected: usize,
        /// Number of parameter values provided by the caller.
        actual: usize,
    },
    /// A numerical parameter value is not finite.
    NonFiniteParameter {
        /// Index of the invalid parameter in the parameter vector.
        index: usize,

        /// Invalid parameter value.
        value: f64,
    },
    /// The prepared MNA contains coefficients that cannot be represented as
    /// real numbers.
    NonRealCoefficients,
    /// Symbolica failed while compiling or evaluating the symbolic expressions.
    Evaluation(
        /// Underlying Symbolica evaluation error.
        EvaluationError,
    ),
    /// Evaluation of the prepared MNA produced a non-finite coefficient.
    NonFiniteBaseCoefficient,
    /// A numerical multiport admittance model is malformed.
    InvalidPortAdmittance {
        /// Description of the invalid port model.
        message: String,
    },
    /// A referenced circuit node is not present in the MNA node map.
    MissingNode(
        /// Name of the missing circuit node.
        String,
    ),
    /// The input voltage of a requested transfer function is exactly zero.
    ZeroInputVoltage {
        /// Input node whose solved voltage is zero.
        node: String,
        /// Frequency at which the zero input voltage was obtained.
        frequency_hz: f64,
    },
    /// The requested AC frequency is invalid.
    InvalidFrequency(
        /// Invalid frequency in hertz.
        f64,
    ),
    /// The assembled numerical MNA system contains a non-finite coefficient.
    NonFiniteSystemCoefficient {
        /// Frequency at which the invalid coefficient was detected.
        frequency_hz: f64,
    },
}

/// Compiles a symbolic MNA system of the form `A(s) = A0 + sC`.
///
/// `parameter_order` defines the exact order in which numerical parameter
/// values must later be passed to [`PreparedNumericMna::instantiate`].
///
/// All symbolic parameters present in the MNA matrix and right-hand side must
/// appear exactly once in `parameter_order`.
///
/// Matrix entries may contain a constant term and a term linear in `s`. Higher
/// powers, negative powers, non-polynomial dependencies, and frequency-
/// dependent right-hand-side entries are rejected. Additional numerical
/// capacitance terms may still be stamped later into the instantiated
/// [`NumericMnaSystem`].
///
/// # Errors
///
/// Returns [`NumericMnaError::InvalidBaseDimensions`] if the MNA matrix is
/// empty or non-square, or if the right-hand side is not a compatible column
/// vector.
///
/// Returns [`NumericMnaError::NonAffineFrequencyDependence`] if a matrix entry
/// is not affine in `s`, or [`NumericMnaError::FrequencyDependentRhs`] if the
/// right-hand side contains `s`.
///
/// Returns [`NumericMnaError::ParameterMismatch`] if `parameter_order` does not
/// contain exactly the symbolic parameters found in the system.
///
/// Returns [`NumericMnaError::NonRealCoefficients`] if the compiled expressions
/// contain non-real coefficients.
///
/// Returns [`NumericMnaError::Evaluation`] if Symbolica cannot compile the
/// expressions.
impl PreparedNumericMna {
    /// Compile a symbolic MNA affine in `s` using an exact parameter order.
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
        let s = Atom::var(symbolica::symbol!("s"));
        let mut base_expressions = Vec::with_capacity(matrix_len);
        let mut capacitance_expressions = Vec::with_capacity(matrix_len);
        for row in 0..rows {
            let matrix_row =
                u32::try_from(row).expect("MNA row count must fit Symbolica matrix indices");
            for column in 0..columns {
                let matrix_column = u32::try_from(column)
                    .expect("MNA column count must fit Symbolica matrix indices");
                let expression = system.a[(matrix_row, matrix_column)].clone();
                let (base, capacitance) = split_affine_in_s(&expression, &s).ok_or_else(|| {
                    NumericMnaError::NonAffineFrequencyDependence {
                        row,
                        column,
                        expression: expression.to_string(),
                    }
                })?;
                base_expressions.push(base);
                capacitance_expressions.push(capacitance);
            }
        }

        let mut rhs_expressions = Vec::with_capacity(rows);
        for row in 0..rows {
            let matrix_row =
                u32::try_from(row).expect("MNA row count must fit Symbolica matrix indices");
            let expression = system.z[(matrix_row, 0)].clone();
            if contains_laplace_variable(&expression) {
                return Err(NumericMnaError::FrequencyDependentRhs {
                    row,
                    expression: expression.to_string(),
                });
            }
            rhs_expressions.push(expression);
        }

        let expressions = base_expressions
            .into_iter()
            .chain(capacitance_expressions)
            .chain(rhs_expressions)
            .collect::<Vec<_>>();

        let mut symbols_by_name = BTreeMap::new();
        for expression in &expressions {
            for symbol in expression.get_all_symbols(false) {
                let symbol = Atom::from(symbol);
                symbols_by_name.entry(symbol.to_string()).or_insert(symbol);
            }
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

    /// Returns the parameter names in the order expected by [`Self::instantiate`].
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Instantiates the prepared MNA system with numerical parameter values.
    ///
    /// `parameter_values` must follow the order returned by
    /// [`Self::parameter_names`].
    ///
    /// The symbolic matrix and right-hand side are evaluated numerically and used
    /// to initialize a new [`NumericMnaSystem`]. Its capacitance matrix contains
    /// the coefficients of `s` from the original netlist and can subsequently be
    /// augmented using [`NumericMnaSystem::stamp_port_admittance`].
    ///
    /// # Errors
    ///
    /// Returns [`NumericMnaError::InvalidParameterCount`] if the number of values
    /// does not match the number of prepared parameters.
    ///
    /// Returns [`NumericMnaError::NonFiniteParameter`] if any parameter is NaN or
    /// infinite.
    ///
    /// Returns [`NumericMnaError::Evaluation`] if the compiled symbolic expressions
    /// cannot be evaluated.
    ///
    /// Returns [`NumericMnaError::NonFiniteBaseCoefficient`] if evaluation produces
    /// a NaN or infinite matrix coefficient.
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
            base: DMatrix::from_row_slice(self.rows, self.rows, &self.outputs[..self.matrix_len]),
            capacitance: DMatrix::from_row_slice(
                self.rows,
                self.rows,
                &self.outputs[self.matrix_len..2 * self.matrix_len],
            ),
            rhs: DVector::from_row_slice(&self.outputs[2 * self.matrix_len..]),
            nodes: self.nodes.clone(),
        })
    }
}

fn split_affine_in_s(expression: &Atom, s: &Atom) -> Option<(Atom, Atom)> {
    let mut base = Atom::num(0);
    let mut capacitance = Atom::num(0);
    let one = Atom::num(1);

    for (power, coefficient) in expression.coefficient_list::<i16>(std::slice::from_ref(s)) {
        if contains_laplace_variable(&coefficient) {
            return None;
        }
        if power == one {
            base += coefficient;
        } else if power == *s {
            capacitance += coefficient;
        } else {
            return None;
        }
    }

    Some((base, capacitance))
}

fn contains_laplace_variable(expression: &Atom) -> bool {
    expression
        .get_all_symbols(false)
        .into_iter()
        .any(|symbol| Atom::from(symbol).to_string() == "s")
}

impl NumericMnaSystem {
    /// Returns the frequency-independent MNA matrix.
    pub fn base_matrix(&self) -> &DMatrix<f64> {
        &self.base
    }
    /// Returns the capacitance matrix of the numerical MNA system.
    pub fn capacitance_matrix(&self) -> &DMatrix<f64> {
        &self.capacitance
    }
    /// Returns the numerical right-hand-side vector.
    pub fn rhs(&self) -> &DVector<f64> {
        &self.rhs
    }

    /// Stamps a numerical multiport admittance model into the MNA system.
    ///
    /// The port model is represented as
    ///
    /// ```text
    /// Y(s) = G + sC
    /// ```
    ///
    /// where `conductance` contains `G` and `capacitance` contains `C`.
    ///
    /// `port_order` defines the row and column ordering of both matrices, while
    /// `connections` maps each port name to a circuit node name.
    ///
    /// Ports connected to ground are omitted according to the reduced MNA
    /// formulation. Multiple physical ports may map to the same circuit node; in
    /// that case their contributions are accumulated into the same MNA entries.
    ///
    /// # Errors
    ///
    /// Returns [`NumericMnaError::InvalidPortAdmittance`] if the port matrices have
    /// invalid dimensions, a port has no connection, a node lies outside the MNA
    /// system, or a coefficient is not finite.
    ///
    /// Returns [`NumericMnaError::MissingNode`] if a referenced circuit node does
    /// not exist in the node map.
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
                .get(node_name)
                .ok_or_else(|| NumericMnaError::MissingNode(node_name.clone()))?;
            let matrix_index = node_number.checked_sub(1);
            if matrix_index.is_some_and(|index| index >= self.base.nrows()) {
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
                self.base[(mna_row, mna_column)] += g;
                self.capacitance[(mna_row, mna_column)] += c;
            }
        }
        Ok(())
    }

    /// Solves the numerical MNA system at one frequency and returns a node voltage.
    ///
    /// The AC system matrix is constructed as
    ///
    /// ```text
    /// A(jω) = A0 + jωC
    /// ```
    ///
    /// with `ω = 2πf`, and the resulting complex linear system is solved using LU
    /// decomposition.
    ///
    /// A frequency of `0.0` performs a DC solution.
    ///
    /// Returns `Ok(None)` if the MNA matrix is singular and the linear system cannot
    /// be solved.
    ///
    /// # Errors
    ///
    /// Returns [`NumericMnaError::InvalidFrequency`] if `frequency_hz` is negative,
    /// NaN, or infinite.
    ///
    /// Returns [`NumericMnaError::MissingNode`] if `node_name` is not present in
    /// the node map.
    ///
    /// Returns [`NumericMnaError::InvalidPortAdmittance`] if the requested node is
    /// ground or lies outside the MNA matrix.
    ///
    /// Returns [`NumericMnaError::NonFiniteSystemCoefficient`] if the assembled
    /// system or resulting node voltage contains a non-finite value.
    pub fn solve_node(
        &self,
        frequency_hz: f64,
        node_name: &str,
    ) -> Result<Option<Complex64>, NumericMnaError> {
        validate_frequency(frequency_hz)?;
        let output_row = self.node_voltage_index(node_name)?;
        let Some(solution) = self.solve_at_frequency(frequency_hz)? else {
            return Ok(None);
        };
        let output = solution[output_row];
        validate_complex_result(output, frequency_hz)?;
        Ok(Some(output))
    }

    /// Solves the numerical MNA system at one frequency and returns `Vout / Vin`.
    ///
    /// `input_node` and `output_node` identify single-ended node voltages relative
    /// to ground. The AC system is factored and solved only once; both voltages
    /// are extracted from the same solution vector.
    ///
    /// A frequency of `0.0` performs a DC solution.
    ///
    /// Returns `Ok(None)` if the MNA matrix is singular and the linear system cannot
    /// be solved.
    ///
    /// # Errors
    ///
    /// Returns [`NumericMnaError::InvalidFrequency`] if `frequency_hz` is negative,
    /// NaN, or infinite.
    ///
    /// Returns [`NumericMnaError::MissingNode`] if either requested node is absent
    /// from the node map.
    ///
    /// Returns [`NumericMnaError::InvalidPortAdmittance`] if either requested node
    /// is ground or lies outside the MNA matrix.
    ///
    /// Returns [`NumericMnaError::ZeroInputVoltage`] if the solved input voltage is
    /// exactly zero.
    ///
    /// Returns [`NumericMnaError::NonFiniteSystemCoefficient`] if the assembled
    /// system, either requested node voltage, or the resulting transfer contains
    /// a non-finite value.
    pub fn solve_transfer(
        &self,
        frequency_hz: f64,
        input_node: &str,
        output_node: &str,
    ) -> Result<Option<Complex64>, NumericMnaError> {
        validate_frequency(frequency_hz)?;
        let input_row = self.node_voltage_index(input_node)?;
        let output_row = self.node_voltage_index(output_node)?;
        let Some(solution) = self.solve_at_frequency(frequency_hz)? else {
            return Ok(None);
        };
        let input = solution[input_row];
        let output = solution[output_row];
        validate_complex_result(input, frequency_hz)?;
        validate_complex_result(output, frequency_hz)?;
        if input == Complex64::new(0.0, 0.0) {
            return Err(NumericMnaError::ZeroInputVoltage {
                node: input_node.to_owned(),
                frequency_hz,
            });
        }
        let transfer = output / input;
        validate_complex_result(transfer, frequency_hz)?;
        Ok(Some(transfer))
    }

    fn node_voltage_index(&self, node_name: &str) -> Result<usize, NumericMnaError> {
        let node_number = self
            .nodes
            .get(node_name)
            .ok_or_else(|| NumericMnaError::MissingNode(node_name.to_owned()))?;
        let row =
            node_number
                .checked_sub(1)
                .ok_or_else(|| NumericMnaError::InvalidPortAdmittance {
                    message: format!("cannot return ground node '{node_name}'"),
                })?;
        if row >= self.base.nrows() {
            return Err(NumericMnaError::InvalidPortAdmittance {
                message: format!("node '{node_name}' is outside the MNA matrix"),
            });
        }
        Ok(row)
    }

    fn solve_at_frequency(
        &self,
        frequency_hz: f64,
    ) -> Result<Option<DVector<Complex64>>, NumericMnaError> {
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let matrix = DMatrix::from_fn(self.base.nrows(), self.base.ncols(), |row, column| {
            Complex64::new(
                self.base[(row, column)],
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
        Ok(Some(solution))
    }
}

fn validate_frequency(frequency_hz: f64) -> Result<(), NumericMnaError> {
    if !frequency_hz.is_finite() || frequency_hz < 0.0 {
        return Err(NumericMnaError::InvalidFrequency(frequency_hz));
    }
    Ok(())
}

fn validate_complex_result(value: Complex64, frequency_hz: f64) -> Result<(), NumericMnaError> {
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(NumericMnaError::NonFiniteSystemCoefficient { frequency_hz });
    }
    Ok(())
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
            Self::NonAffineFrequencyDependence {
                row,
                column,
                expression,
            } => write!(
                formatter,
                "MNA entry ({row}, {column}) is not affine in s: {expression}"
            ),
            Self::FrequencyDependentRhs { row, expression } => write!(
                formatter,
                "MNA right-hand-side entry {row} depends on s: {expression}"
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
                formatter.write_str("prepared MNA contains non-real coefficients")
            }
            Self::Evaluation(error) => write!(formatter, "could not evaluate prepared MNA: {error}"),
            Self::NonFiniteBaseCoefficient => {
                formatter.write_str("prepared MNA evaluation produced a non-finite coefficient")
            }
            Self::InvalidPortAdmittance { message } => {
                write!(formatter, "invalid port admittance: {message}")
            }
            Self::MissingNode(node) => write!(formatter, "MNA node '{node}' is missing"),
            Self::ZeroInputVoltage { node, frequency_hz } => write!(
                formatter,
                "transfer-function input node '{node}' is zero at {frequency_hz:.6e} Hz"
            ),
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

    use nalgebra::{DMatrix, DVector};
    use ndarray::array;
    use num_complex::Complex64;
    use symbolica::domains::atom::AtomField;
    use symbolica::prelude::{Matrix, parse};

    use super::{NumericMnaError, NumericMnaSystem, PreparedNumericMna};
    use crate::mna::MnaResult;
    use crate::spice2cir::NodeMap;

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
            nodes: NodeMap::from_map(HashMap::from([
                    ("VIN".to_owned(), 1),
                    ("VOUT".to_owned(), 2),
                    ("VSS".to_owned(), 0),
                ])),
        }
    }

    fn rc_low_pass_with_netlist_capacitance() -> MnaResult {
        let mut system = rc_low_pass();
        system.a[(1, 1)] += parse!("s*c");
        system
    }

    fn two_node_numeric_system(base: DMatrix<f64>, rhs: [f64; 2]) -> NumericMnaSystem {
        NumericMnaSystem {
            base,
            capacitance: DMatrix::zeros(2, 2),
            rhs: DVector::from_row_slice(&rhs),
            nodes: NodeMap::from_map(HashMap::from([
                ("VIN".to_owned(), 1),
                ("VOUT".to_owned(), 2),
                ("VSS".to_owned(), 0),
                ("OUTSIDE".to_owned(), 3),
            ])),
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
    fn solves_a_parameterized_rc_low_pass_with_capacitance_from_the_netlist() {
        let mut prepared =
            PreparedNumericMna::new(&rc_low_pass_with_netlist_capacitance(), &["r", "c"])
                .unwrap();
        let resistance = 1.0e3;
        let capacitance = 1.0e-9;
        let mut system = prepared.instantiate(&[resistance, capacitance]).unwrap();

        assert_eq!(system.capacitance_matrix()[(1, 1)], capacitance);
        let pole_hz = 1.0 / (2.0 * std::f64::consts::PI * resistance * capacitance);
        let at_pole = system.solve_node(pole_hz, "VOUT").unwrap().unwrap();
        assert!((at_pole.norm() - 2.0_f64.sqrt().recip()).abs() < 1.0e-12);
        assert!((at_pole.arg().to_degrees() + 45.0).abs() < 1.0e-10);

        stamp_output_capacitance(&mut system, capacitance).unwrap();
        assert_eq!(system.capacitance_matrix()[(1, 1)], 2.0 * capacitance);
        let combined_pole_hz =
            1.0 / (2.0 * std::f64::consts::PI * resistance * 2.0 * capacitance);
        let at_combined_pole = system
            .solve_node(combined_pole_hz, "VOUT")
            .unwrap()
            .unwrap();
        assert!((at_combined_pole.norm() - 2.0_f64.sqrt().recip()).abs() < 1.0e-12);
    }

    #[test]
    fn solves_a_node_transfer_independently_of_input_amplitude() {
        let mut symbolic = rc_low_pass_with_netlist_capacitance();
        symbolic.z[(2, 0)] = parse!("2");
        let mut prepared = PreparedNumericMna::new(&symbolic, &["r", "c"]).unwrap();
        let resistance = 1.0e3;
        let capacitance = 1.0e-9;
        let system = prepared.instantiate(&[resistance, capacitance]).unwrap();

        let input = system.solve_node(0.0, "VIN").unwrap().unwrap();
        let output = system.solve_node(0.0, "VOUT").unwrap().unwrap();
        let dc_transfer = system.solve_transfer(0.0, "VIN", "VOUT").unwrap().unwrap();
        assert_eq!(input, Complex64::new(2.0, 0.0));
        assert_eq!(output, Complex64::new(2.0, 0.0));
        assert!((dc_transfer - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);

        let pole_hz = 1.0 / (2.0 * std::f64::consts::PI * resistance * capacitance);
        let at_pole = system
            .solve_transfer(pole_hz, "VIN", "VOUT")
            .unwrap()
            .unwrap();
        assert!((at_pole.norm() - 2.0_f64.sqrt().recip()).abs() < 1.0e-12);
        assert!((at_pole.arg().to_degrees() + 45.0).abs() < 1.0e-10);
    }

    #[test]
    fn validates_transfer_nodes_input_voltage_and_singular_systems() {
        let system = two_node_numeric_system(DMatrix::identity(2, 2), [1.0, 1.0]);
        assert!(matches!(
            system.solve_transfer(-1.0, "VIN", "VOUT"),
            Err(NumericMnaError::InvalidFrequency(value)) if value == -1.0
        ));
        assert!(matches!(
            system.solve_transfer(0.0, "MISSING", "VOUT"),
            Err(NumericMnaError::MissingNode(node)) if node == "MISSING"
        ));
        assert!(matches!(
            system.solve_transfer(0.0, "VIN", "MISSING"),
            Err(NumericMnaError::MissingNode(node)) if node == "MISSING"
        ));
        assert!(matches!(
            system.solve_transfer(0.0, "VSS", "VOUT"),
            Err(NumericMnaError::InvalidPortAdmittance { .. })
        ));
        assert!(matches!(
            system.solve_transfer(0.0, "VIN", "VSS"),
            Err(NumericMnaError::InvalidPortAdmittance { .. })
        ));
        assert!(matches!(
            system.solve_transfer(0.0, "OUTSIDE", "VOUT"),
            Err(NumericMnaError::InvalidPortAdmittance { .. })
        ));

        let zero_input = two_node_numeric_system(DMatrix::identity(2, 2), [0.0, 1.0]);
        assert!(matches!(
            zero_input.solve_transfer(1.0e3, "VIN", "VOUT"),
            Err(NumericMnaError::ZeroInputVoltage { node, frequency_hz })
                if node == "VIN" && frequency_hz == 1.0e3
        ));

        let singular = two_node_numeric_system(DMatrix::zeros(2, 2), [1.0, 1.0]);
        assert_eq!(singular.solve_transfer(1.0e3, "VIN", "VOUT").unwrap(), None);
    }

    #[test]
    fn rejects_non_affine_frequency_dependence_and_frequency_dependent_rhs() {
        let mut quadratic = rc_low_pass();
        quadratic.a[(1, 1)] += parse!("s^2*c");
        assert!(matches!(
            PreparedNumericMna::new(&quadratic, &["r", "c"]),
            Err(NumericMnaError::NonAffineFrequencyDependence {
                row: 1,
                column: 1,
                ..
            })
        ));

        let mut reciprocal = rc_low_pass();
        reciprocal.a[(1, 1)] += parse!("1/s");
        assert!(matches!(
            PreparedNumericMna::new(&reciprocal, &["r"]),
            Err(NumericMnaError::NonAffineFrequencyDependence { .. })
        ));

        let mut rational = rc_low_pass();
        rational.a[(1, 1)] += parse!("1/(1+s)");
        assert!(matches!(
            PreparedNumericMna::new(&rational, &["r"]),
            Err(NumericMnaError::NonAffineFrequencyDependence { .. })
        ));

        let mut hidden = rc_low_pass();
        hidden.a[(1, 1)] += parse!("f(s)");
        assert!(matches!(
            PreparedNumericMna::new(&hidden, &["r"]),
            Err(NumericMnaError::NonAffineFrequencyDependence { .. })
        ));

        let mut frequency_dependent_rhs = rc_low_pass();
        frequency_dependent_rhs.z[(0, 0)] = parse!("s");
        assert!(matches!(
            PreparedNumericMna::new(&frequency_dependent_rhs, &["r"]),
            Err(NumericMnaError::FrequencyDependentRhs { row: 0, .. })
        ));
    }

    #[test]
    fn includes_capacitance_only_parameters_in_parameter_validation() {
        assert!(matches!(
            PreparedNumericMna::new(&rc_low_pass_with_netlist_capacitance(), &["r"]),
            Err(NumericMnaError::ParameterMismatch { expected, .. })
                if expected == vec!["c".to_owned(), "r".to_owned()]
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

    #[test]
    fn accumulates_multiple_capacitance_stamps_on_a_shared_node() {
        let mut system = two_node_numeric_system(DMatrix::zeros(2, 2), [0.0, 0.0]);

        stamp_output_capacitance(&mut system, 2.0).unwrap();
        stamp_output_capacitance(&mut system, 3.0).unwrap();

        assert_eq!(system.capacitance_matrix()[(1, 1)], 5.0);
    }
}
