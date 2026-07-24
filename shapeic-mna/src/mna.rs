use ndarray::ArrayView2;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use symbolica::domains::atom::AtomField;
use symbolica::prelude::{Matrix, parse};

use crate::spice_parser::{NodeMap, SpiceError, spice_parser};
use crate::symmna::{SmnaError, Vector, smna};

#[derive(Debug)]
pub struct MnaResult {
    pub report: String,
    pub a: Matrix<AtomField>,
    pub x: Vector,
    pub z: Matrix<AtomField>,
    pub nodes: NodeMap,
}

impl MnaResult {
    pub fn node_name_for_variable(&self, variable: &str) -> Option<&str> {
        let node_number = variable.strip_prefix('v')?.parse::<usize>().ok()?;

        self.nodes
            .nodes
            .iter()
            .find_map(|(name, number)| (*number == node_number).then_some(name.as_str()))
    }

    pub fn variable_for_node_name(&self, node_name: &str) -> Option<String> {
        let node_number = self.nodes.nodes.get(node_name)?;

        if *node_number == 0 {
            None
        } else {
            Some(format!("v{node_number}"))
        }
    }

    pub fn node_variables(&self) -> Vec<NodeVariable> {
        let mut variables = self
            .nodes
            .nodes
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeVariable {
    pub variable: String,
    pub node_name: String,
    pub node_number: usize,
}

#[derive(Debug)]
pub struct MnaSolveResult {
    pub variables: Vector,
    pub solution: Matrix<AtomField>,
}

#[derive(Debug)]
pub enum MnaError {
    Io(std::io::Error),
    SpiceConversion(SpiceError),
    SymMna(SmnaError),
    PythonSolve(String),
    MissingNode(String),
    DimensionMismatch { message: String },
    SolveFailed { message: String },
    InvalidPortAdmittance { message: String },
}

impl From<std::io::Error> for MnaError {
    fn from(error: std::io::Error) -> Self {
        MnaError::Io(error)
    }
}

pub fn mna(spice_dir: &Path, output_dir: &Path, design_name: &str) -> Result<MnaResult, MnaError> {
    let nodes =
        spice_parser(spice_dir, output_dir, design_name).map_err(MnaError::SpiceConversion)?;
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
            message: format!("z must be a column matrix, but i has {} columns", z.ncols()),
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

/// Stamp a complete physical port model as `G + sC` into an existing MNA matrix.
///
/// `connections` maps each name in `port_order` to a circuit node name. A node
/// mapped to the netlist ground is omitted in the usual reduced MNA form.
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
            .nodes
            .get(node_name)
            .copied()
            .ok_or_else(|| MnaError::MissingNode(node_name.clone()))?;
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
    let nodes = NodeMap {
        nodes: HashMap::from([("P".to_owned(), 1), ("VSS".to_owned(), 0)]),
    };
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

    assert_eq!(a[(0, 0)], parse!("2+3*s"));
}
