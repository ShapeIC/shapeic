use std::fs;
use std::path::Path;
use symbolica::prelude::Matrix;
#[cfg(test)]
use symbolica::prelude::parse;
use symbolica::domains::atom::AtomField;

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
    DimensionMismatch{
        message: String,
    },
    SolveFailed {
        message: String,
    },
}

impl From<std::io::Error> for MnaError {
    fn from(error: std::io::Error) -> Self {
        MnaError::Io(error)
    }
}

pub fn mna(spice_dir: &Path, output_dir: &Path, design_name: &str) -> Result<MnaResult, MnaError> {
    let nodes = spice_parser(spice_dir, output_dir, design_name).map_err(MnaError::SpiceConversion)?;
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

pub fn mna_solve(a: &Matrix<AtomField>, x:&Vector, z: &Matrix<AtomField>) -> Result<MnaSolveResult, MnaError> {
    let rows = a.nrows();
    let cols = a.ncols();

    if rows != cols {
        return Err(MnaError::DimensionMismatch {message: format!("A must be square, but its dimensions are {rows}x{cols}")});
    }

    if z.ncols() != 1 {
        return Err(MnaError::DimensionMismatch {message: format!("z must be a column matrix, but i has {} columns", z.ncols())});
    }

    // A y z deben tener el mismo número de filas.
    if z.nrows() != rows {
        return Err(MnaError::DimensionMismatch {message: format!("A has {rows} rows, but z has {} rows",z.nrows())});
    }

    let solution = a
        .solve(z)
        .map_err(|error| MnaError::SolveFailed {
            message: error.to_string(),
        })?;

    Ok(MnaSolveResult {
        variables: x.clone(),
        solution,
    })
}

#[test]
fn solves_basic_netlist_from_smna() {
    let content = "R1 1 0 1\nI1 1 0 1\n";

    let system = smna(content)
        .expect("SMNA should build the MNA system");

    let result = mna_solve(
        &system.a,
        &system.x,
        &system.z,
    )
    .expect("MNA system should be solvable");

    assert_eq!(result.variables.len(), 1);
    assert_eq!(result.variables[0], parse!("v1"));

    assert_eq!(result.solution.nrows(), 1);
    assert_eq!(result.solution.ncols(), 1);

    assert_eq!(
        result.solution[(0, 0)],
        parse!("-1")
    );
}
