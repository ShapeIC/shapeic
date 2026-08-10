//! SPICE netlist preprocessing and node mapping.
//!
//! This module converts SPICE netlists with named nodes into the
//! numeric-node representation expected by the symbolic MNA implementation.
//!
//! During parsing, each unique node name is assigned a numeric identifier.
//! The `vss` net is treated as ground and mapped to node `0`.
use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::path::Path;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone)]
/// Mapping between SPICE net names and their numeric node identifiers.
///
/// Non-ground nodes are numbered sequentially starting from `1`.
/// The `vss` net is treated as ground and assigned node `0`.
pub struct NodeMap {
    nodes: HashMap<String, usize>,
}

impl NodeMap {
    /// Returns the numeric identifier associated with a SPICE net name.
    pub fn get(&self, net: &str) -> Option<usize> {
        self.nodes.get(net).copied()
    }

    /// Returns the complete mapping from SPICE net names to numeric node identifiers.
    pub fn nodes(&self) -> &HashMap<String, usize> {
        &self.nodes
    }

    pub(crate) fn from_map(nodes: HashMap<String, usize>) -> Self {
        Self { nodes }
    }
}

#[derive(Debug)]
/// Errors that may occur while preprocessing a SPICE netlist.
pub enum SpiceError {
    /// An I/O operation failed while reading or writing the netlist.
    Io(std::io::Error),
}

impl From<std::io::Error> for SpiceError {
    fn from(error: std::io::Error) -> Self {
        SpiceError::Io(error)
    }
}

/// Converts a SPICE netlist with named nodes into a numeric-node netlist.
///
/// The input netlist is read from:
///
/// ```text
/// <spice_dir>/<filename>.spice
/// ```
///
/// and the converted netlist is written to:
///
/// ```text
/// <output_dir>/<filename>.cir
/// ```
///
/// Node names appearing in supported circuit elements are replaced by
/// sequential numeric identifiers. The `vss` net is mapped to node `0`.
///
/// The returned [`NodeMap`] preserves the correspondence between the
/// original node names and their assigned numeric identifiers.
///
/// # Errors
///
/// Returns [`SpiceError::Io`] if the input file cannot be read, the output
/// directory or file cannot be created, or writing the converted netlist fails.
pub fn spice2cir(
    spice_dir: &Path,
    output_dir: &Path,
    filename: &str,
) -> Result<NodeMap, SpiceError> {
    let spice_path = spice_dir.join(format!("{filename}.spice"));
    let output_path = output_dir.join(format!("{filename}.cir"));

    create_dir_all(output_dir)?;

    let spice_file = File::open(spice_path)?;
    let reader = BufReader::new(spice_file);

    let mut output_file = File::create(output_path)?;

    let mut nodes: HashMap<String, usize> = HashMap::new();
    let mut node_num: usize = 1;

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty(){
            continue;
        }
        let first_char = trimmed.chars().next().unwrap();

        if first_char == '*' || first_char == '.' {
            continue;
        }

        let mut params: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();

        match first_char {
            'R' | 'C' | 'L' => {
                replace_node(&mut params, 1, &mut nodes, &mut node_num);
                replace_node(&mut params, 2, &mut nodes, &mut node_num);

                writeln!(output_file, "{}", params[..4].join(" "))?;
            }

            'G' => {
                replace_node(&mut params, 1, &mut nodes, &mut node_num);
                replace_node(&mut params, 2, &mut nodes, &mut node_num);
                replace_node(&mut params, 3, &mut nodes, &mut node_num);
                replace_node(&mut params, 4, &mut nodes, &mut node_num);

                writeln!(output_file, "{}", params[..6].join(" "))?;
            }

            'V' | 'I' => {
                replace_node(&mut params, 1, &mut nodes, &mut node_num);
                replace_node(&mut params, 2, &mut nodes, &mut node_num);

                writeln!(output_file, "{}", params[..4].join(" "))?;
            }

            _ => {
                writeln!(output_file, "{line}")?;
            }
        }
    }

    Ok(NodeMap { nodes })
}

fn replace_node(
    params: &mut [String],
    index: usize,
    nodes: &mut HashMap<String, usize>,
    node_num: &mut usize,
) {
    let net = params[index].clone();

    let is_ground = net.eq_ignore_ascii_case("vss");

    let number = if is_ground {
        0
    } else {
        match nodes.get(&net) {
            Some(existing_number) => *existing_number,
            None => {
                let new_number = *node_num;
                nodes.insert(net.clone(), new_number);
                *node_num += 1;
                new_number
            }
        }
    };

    if is_ground {
        nodes.entry(net).or_insert(0);
    }

    params[index] = number.to_string();
}
