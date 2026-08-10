//! Modified nodal analysis construction, solution, and physical-model stamping.
//!
//! This module provides the high-level MNA interface used by ShapeIC.
//! It converts SPICE netlists into symbolic MNA systems, solves assembled
//! systems, maps symbolic voltage variables back to circuit node names,
//! and stamps physical multiport admittance models into existing MNA matrices.
use ndarray::ArrayView2;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::collections::HashMap;
use symbolica::domains::atom::AtomField;
use symbolica::prelude::{Matrix, parse};

use crate::spice2cir::{NodeMap, SpiceError, spice2cir};
use crate::symmna::{SmnaError, Vector, smna};

/// Symbolic MNA system generated from a SPICE netlist.
///
/// The system follows the form
///
/// ```text
/// A x = z
/// ```
///
/// and additionally preserves the mapping between the original SPICE
/// node names and their numeric MNA representation.
#[derive(Debug)]
pub struct MnaResult {
    /// Human-readable report describing the generated MNA system.
    pub report: String,

    /// Symbolic MNA system matrix.
    pub a: Matrix<AtomField>,

    /// Ordered vector of MNA unknowns.
    pub x: Vector,

    /// Right-hand-side excitation vector.
    pub z: Matrix<AtomField>,

    /// Mapping between original SPICE node names and numeric node identifiers.
    pub nodes: NodeMap,
}

impl MnaResult {
    /// Returns the SPICE node name corresponding to an MNA voltage variable.
    ///
    /// For example, if `v3` corresponds to the SPICE node `vout`,
    /// this method returns `Some("vout")`.
    pub fn node_name_for_variable(&self, variable: &str) -> Option<&str> {
        let node_number = variable.strip_prefix('v')?.parse::<usize>().ok()?;

        self.nodes
            .nodes()
            .iter()
            .find_map(|(name, number)| {
                (*number == node_number).then_some(name.as_str())
            })
    }
    /// Returns the MNA voltage variable associated with a SPICE node name.
    ///
    /// Ground nodes return `None` because ground is omitted from the
    /// reduced MNA voltage vector.
    pub fn variable_for_node_name(&self, node_name: &str) -> Option<String> {
        let node_number = self.nodes.get(node_name)?;

        if node_number == 0 {
            None
        } else {
            Some(format!("v{node_number}"))
        }
    }

    /// Returns all non-ground node variables ordered by numeric node identifier.
    pub fn node_variables(&self) -> Vec<NodeVariable> {
        let mut variables = self
            .nodes
            .nodes()
            .iter()
            .filter_map(|(node_name, node_number)| {
                if *node_number == 0 {
                    None
                } else {
                    Some(NodeVariable {
                        variable: format!("v{node_number}"),
                        node_name: node_name.clone(),
                        node_number: *node_number,
                    })
                }
            })
            .collect::<Vec<_>>();

        variables.sort_by_key(|variable| variable.node_number);
        variables
    }
}

/// Association between an MNA voltage variable and its original SPICE node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeVariable {
    /// Symbolic MNA voltage variable, such as `v1` or `v3`.
    pub variable: String,

    /// Original SPICE node name.
    pub node_name: String,

    /// Numeric node identifier assigned during SPICE preprocessing.
    pub node_number: usize,
}

/// Solution of an assembled MNA system.
#[derive(Debug)]
pub struct MnaSolveResult {
    /// Ordered MNA variables corresponding to the solution entries.
    pub variables: Vector,

    /// Symbolic solution vector.
    pub solution: Matrix<AtomField>,
}

/// Errors that may occur while building, modifying, or solving an MNA system.
#[derive(Debug)]
pub enum MnaError {
    /// An I/O operation failed.
    Io(
        /// Underlying I/O error.
        std::io::Error,
    ),

    /// SPICE preprocessing failed.
    SpiceConversion(
        /// Underlying SPICE conversion error.
        SpiceError,
    ),

    /// Symbolic MNA generation failed.
    SymMna(
        /// Underlying symbolic MNA error.
        SmnaError,
    ),

    /// A referenced circuit node could not be found.
    MissingNode {
        /// Name of the missing node.
        node: String,
    },

    /// Matrix or vector dimensions are incompatible.
    DimensionMismatch {
        /// Description of the dimension mismatch.
        message: String,
    },

    /// The symbolic linear system could not be solved.
    SolveFailed {
        /// Description of the solver failure.
        message: String,
    },

    /// A physical multiport admittance model is invalid.
    InvalidPortAdmittance {
        /// Description of the invalid admittance model.
        message: String,
    },
}

impl From<std::io::Error> for MnaError {
    fn from(error: std::io::Error) -> Self {
        MnaError::Io(error)
    }
}

/// Builds a symbolic MNA system from a SPICE netlist.
///
/// The netlist `<design_name>.spice` is read from `spice_dir`,
/// converted to the numeric-node `.cir` representation, and passed
/// to the symbolic MNA generator.
///
/// The generated `.cir` file is written to `output_dir`.
///
/// # Errors
///
/// Returns [`MnaError::SpiceConversion`] if the SPICE netlist cannot be
/// converted, [`MnaError::Io`] if the converted netlist cannot be read,
/// or [`MnaError::SymMna`] if symbolic MNA generation fails.
pub fn mna(spice_dir: &Path, output_dir: &Path, design_name: &str) -> Result<MnaResult, MnaError> {
    let nodes =
        spice2cir(spice_dir, output_dir, design_name).map_err(MnaError::SpiceConversion)?;
    let input_path = output_dir.join(format!("{design_name}.cir"));
    let content = fs::read_to_string(input_path)?;

    let symmna_output = smna(&content).map_err(MnaError::SymMna)?;

    Ok(MnaResult {
        report: symmna_output.report,
        a: symmna_output.a,
        x: symmna_output.x,
        z: symmna_output.z,
        nodes,
    })
}

/// Solves an assembled symbolic MNA system.
///
/// Solves
///
/// ```text
/// A x = z
/// ```
///
/// using Symbolica's symbolic linear solver.
///
/// # Errors
///
/// Returns [`MnaError::DimensionMismatch`] if `A` is not square, if `z`
/// is not a column vector, or if their row counts differ.
///
/// Returns [`MnaError::SolveFailed`] if the symbolic system cannot be solved.
pub fn mna_solve(
    a: &Matrix<AtomField>,
    x: &Vector,
    z: &Matrix<AtomField>,
) -> Result<MnaSolveResult, MnaError> {
    let rows = a.nrows();
    let cols = a.ncols();

    if rows != cols {
        return Err(MnaError::DimensionMismatch {
            message: format!("A must be square, but its dimensions are {rows}x{cols}"),
        });
    }

    if z.ncols() != 1 {
        return Err(MnaError::DimensionMismatch {
            message: format!("z must be a column matrix, but it has {} columns", z.ncols()),
        });
    }

    // A y z deben tener el mismo número de filas.
    if z.nrows() != rows {
        return Err(MnaError::DimensionMismatch {
            message: format!("A has {rows} rows, but z has {} rows", z.nrows()),
        });
    }

    let solution = a.solve(z).map_err(|error| MnaError::SolveFailed {
        message: error.to_string(),
    })?;

    Ok(MnaSolveResult {
        variables: x.clone(),
        solution,
    })
}

/// Stamps a physical multiport admittance model into an existing MNA matrix.
///
/// The physical model is represented as
///
/// ```text
/// Y(s) = G + sC
/// ```
///
/// where `G` and `C` are square port-admittance matrices.
///
/// `port_order` defines the row and column order of the physical matrices,
/// while `connections` maps each physical port to a circuit node name.
///
/// Ports connected to the netlist ground are omitted according to the
/// reduced MNA formulation.
///
/// # Errors
///
/// Returns [`MnaError::InvalidPortAdmittance`] if the conductance and
/// capacitance matrices do not match the number of ports, if a physical
/// port has no circuit connection, or if a matrix coefficient is not finite.
///
/// Returns [`MnaError::MissingNode`] if a circuit node referenced by
/// `connections` is not present in the [`NodeMap`].
pub fn stamp_port_admittance(
    a: &mut Matrix<AtomField>,
    nodes: &NodeMap,
    port_order: &[String],
    connections: &BTreeMap<String, String>,
    conductance: ArrayView2<'_, f64>,
    capacitance: ArrayView2<'_, f64>,
) -> Result<(), MnaError> {
    let port_count = port_order.len();
    let expected = [port_count, port_count];
    if conductance.shape() != expected || capacitance.shape() != expected {
        return Err(MnaError::InvalidPortAdmittance {
            message: format!("G and C must both have shape {expected:?}"),
        });
    }
    if connections.len() != port_count {
        return Err(MnaError::InvalidPortAdmittance {
            message: "each physical port must have exactly one circuit-node connection".into(),
        });
    }
    let mut matrix_indices = Vec::with_capacity(port_count);
    for port in port_order {
        let node_name = connections
            .get(port)
            .ok_or_else(|| MnaError::InvalidPortAdmittance {
                message: format!("physical port '{port}' has no circuit-node connection"),
            })?;
        let node_number = nodes
            .get(node_name)
            .ok_or_else(|| MnaError::MissingNode {
                node: node_name.clone(),
            })?;
        matrix_indices.push(if node_number == 0 {
            None
        } else {
            Some(
                u32::try_from(node_number - 1).map_err(|_| MnaError::InvalidPortAdmittance {
                    message: format!("node number for '{node_name}' does not fit the MNA index"),
                })?,
            )
        });
    }
    let s = parse!("s");
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
                return Err(MnaError::InvalidPortAdmittance {
                    message: format!("non-finite coefficient at ({row}, {column})"),
                });
            }
            if g == 0.0 && c == 0.0 {
                continue;
            }
            let g_atom = parse!(format!("{g:.17e}").as_str());
            let c_atom = parse!(format!("{c:.17e}").as_str());
            a[(mna_row, mna_column)] += g_atom + &s * c_atom;
        }
    }
    Ok(())
}

#[test]
fn solves_basic_netlist_from_smna() {
    let content = "R1 1 0 1\nI1 1 0 1\n";

    let system = smna(content).expect("SMNA should build the MNA system");

    let result = mna_solve(&system.a, &system.x, &system.z).expect("MNA system should be solvable");

    assert_eq!(result.variables.len(), 1);
    assert_eq!(result.variables[0], parse!("v1"));

    assert_eq!(result.solution.nrows(), 1);
    assert_eq!(result.solution.ncols(), 1);

    assert_eq!(result.solution[(0, 0)], parse!("-1"));
}

#[test]
fn stamps_a_complete_port_admittance_with_ground_reduction() {
    use std::collections::HashMap;

    use ndarray::array;

    let mut a = Matrix::new(1, 1, AtomField::new());
    let nodes = NodeMap::from_map(HashMap::from([
        ("P".to_owned(), 1),
        ("VSS".to_owned(), 0),
    ]));
    let ports = vec!["P".to_owned(), "N".to_owned()];
    let connections = BTreeMap::from([
        ("P".to_owned(), "P".to_owned()),
        ("N".to_owned(), "VSS".to_owned()),
    ]);
    let conductance = array![[2.0, -2.0], [-2.0, 2.0]];
    let capacitance = array![[3.0, -3.0], [-3.0, 3.0]];

    stamp_port_admittance(
        &mut a,
        &nodes,
        &ports,
        &connections,
        conductance.view(),
        capacitance.view(),
    )
    .expect("port model should stamp");

    let expected_g = parse!(format!("{:.17e}", 2.0_f64).as_str());
    let expected_c = parse!(format!("{:.17e}", 3.0_f64).as_str());
    let s = parse!("s");

    let expected = expected_g + &s * expected_c;

    assert_eq!(a[(0, 0)], expected);
}
