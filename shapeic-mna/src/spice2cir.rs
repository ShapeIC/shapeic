//! SPICE netlist preprocessing and node mapping.
//!
//! This module converts SPICE netlists with named nodes into the
//! numeric-node representation expected by the symbolic MNA implementation.
//!
//! During parsing, each unique node name is assigned a numeric identifier.
//! The standard node `0` and the ShapeIC convention `VSS` are both treated as
//! ground and mapped to node `0`.

use std::collections::HashMap;
use std::fs::{self, create_dir_all};
use std::io::Write;
use std::path::Path;

/// Mapping between SPICE net names and their numeric node identifiers.
///
/// Non-ground nodes are numbered sequentially starting from `1`. The standard
/// node `0` and `VSS` (case-insensitive) are treated as ground.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Errors that may occur while preprocessing a SPICE netlist.
#[derive(Debug)]
pub enum SpiceError {
    /// An I/O operation failed while reading or writing the netlist.
    Io(std::io::Error),
    /// A supported element contains an unexpected number of tokens.
    BadTokenCount {
        /// One-based line number in the source netlist.
        line: usize,
        /// Contents of the malformed line.
        content: String,
        /// Exact number of tokens required by the supported syntax.
        expected: usize,
        /// Number of tokens found.
        actual: usize,
    },
    /// The netlist contains an element outside the supported linear subset.
    UnsupportedElement {
        /// One-based line number in the source netlist.
        line: usize,
        /// Contents of the unsupported line.
        content: String,
    },
}

impl From<std::io::Error> for SpiceError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Converts SPICE source text into the numeric-node netlist consumed by MNA.
///
/// Empty lines, comments beginning with `*`, and directives beginning with `.`
/// are omitted. Element designators are case-insensitive. The supported linear
/// subset is `R`, `L`, `C`, `V`, `I`, `O`, `E`, `G`, `F`, `H`, and `K`, matching
/// the symbolic MNA implementation.
///
/// Values must occupy exactly one token. Consequently, an independent source is
/// written as `Vname nplus nminus value`; simulator-specific forms such as
/// `AC 1` are not interpreted here.
///
/// # Errors
///
/// Returns [`SpiceError::BadTokenCount`] for malformed supported elements and
/// [`SpiceError::UnsupportedElement`] for element types outside the supported
/// subset.
pub fn spice2cir_text(source: &str) -> Result<(String, NodeMap), SpiceError> {
    let mut nodes = HashMap::new();
    let mut next_node = 1;
    let mut converted = Vec::new();

    for (line_index, source_line) in source.lines().enumerate() {
        let line = source_line.trim();
        if line.is_empty() || line.starts_with('*') || line.starts_with('.') {
            continue;
        }

        let mut tokens = line
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let kind = tokens[0]
            .chars()
            .next()
            .expect("a non-empty token must contain a character")
            .to_ascii_uppercase();
        let (expected, node_indices): (usize, &[usize]) = match kind {
            'R' | 'L' | 'C' | 'V' | 'I' => (4, &[1, 2]),
            'O' => (4, &[1, 2, 3]),
            'E' | 'G' => (6, &[1, 2, 3, 4]),
            'F' | 'H' => (5, &[1, 2]),
            'K' => (4, &[]),
            _ => {
                return Err(SpiceError::UnsupportedElement {
                    line: line_index + 1,
                    content: line.to_owned(),
                });
            }
        };

        if tokens.len() != expected {
            return Err(SpiceError::BadTokenCount {
                line: line_index + 1,
                content: line.to_owned(),
                expected,
                actual: tokens.len(),
            });
        }

        for &node_index in node_indices {
            replace_node(&mut tokens, node_index, &mut nodes, &mut next_node);
        }
        converted.push(tokens.join(" "));
    }

    let mut content = converted.join("\n");
    if !content.is_empty() {
        content.push('\n');
    }
    Ok((content, NodeMap { nodes }))
}

/// Converts a SPICE file with named nodes into a numeric-node `.cir` file.
///
/// The input is read from `<spice_dir>/<filename>.spice` and the converted
/// netlist is written to `<output_dir>/<filename>.cir`. This filesystem wrapper
/// preserves the historical ShapeIC API and delegates parsing to
/// [`spice2cir_text`].
///
/// # Errors
///
/// Returns [`SpiceError`] when the input cannot be read, the netlist is invalid,
/// or the output cannot be created.
pub fn spice2cir(
    spice_dir: &Path,
    output_dir: &Path,
    filename: &str,
) -> Result<NodeMap, SpiceError> {
    let spice_path = spice_dir.join(format!("{filename}.spice"));
    let output_path = output_dir.join(format!("{filename}.cir"));
    let source = fs::read_to_string(spice_path)?;
    let (converted, nodes) = spice2cir_text(&source)?;

    create_dir_all(output_dir)?;
    let mut output = fs::File::create(output_path)?;
    output.write_all(converted.as_bytes())?;

    Ok(nodes)
}

fn replace_node(
    tokens: &mut [String],
    index: usize,
    nodes: &mut HashMap<String, usize>,
    next_node: &mut usize,
) {
    let net = tokens[index].clone();
    let number = if is_ground(&net) {
        nodes.entry(net).or_insert(0);
        0
    } else if let Some(number) = nodes.get(&net) {
        *number
    } else {
        let number = *next_node;
        nodes.insert(net, number);
        *next_node += 1;
        number
    };
    tokens[index] = number.to_string();
}

fn is_ground(net: &str) -> bool {
    net == "0" || net.eq_ignore_ascii_case("vss")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{spice2cir, spice2cir_text, SpiceError};

    #[test]
    fn converts_text_in_memory_and_maps_zero_and_vss_to_ground() {
        let source = "\
* Mixed-case linear circuit
v1 VIN 0 1
r1 VIN VOUT resistance
Cload VOUT vss 1e-12
.ac dec 10 1 1e9
.end
";

        let (converted, nodes) = spice2cir_text(source).unwrap();

        assert_eq!(converted, "v1 1 0 1\nr1 1 2 resistance\nCload 2 0 1e-12\n");
        assert_eq!(nodes.get("VIN"), Some(1));
        assert_eq!(nodes.get("VOUT"), Some(2));
        assert_eq!(nodes.get("0"), Some(0));
        assert_eq!(nodes.get("vss"), Some(0));
    }

    #[test]
    fn converts_nodes_for_every_supported_controlled_element() {
        let source = "\
O1 INP INN OUT
E1 OUT 0 INP INN gain
G1 OUT 0 INP INN gm
V1 CTRL 0 1
F1 OUT 0 V1 gain
H1 OUT 0 V1 resistance
L1 A 0 inductance
L2 B 0 inductance
K1 L1 L2 coupling
";

        let (converted, nodes) = spice2cir_text(source).unwrap();

        assert_eq!(nodes.get("INP"), Some(1));
        assert_eq!(nodes.get("INN"), Some(2));
        assert_eq!(nodes.get("OUT"), Some(3));
        assert_eq!(nodes.get("CTRL"), Some(4));
        assert_eq!(nodes.get("A"), Some(5));
        assert_eq!(nodes.get("B"), Some(6));
        assert!(converted.contains("O1 1 2 3"));
        assert!(converted.contains("E1 3 0 1 2 gain"));
        assert!(converted.contains("F1 3 0 V1 gain"));
        assert!(converted.contains("K1 L1 L2 coupling"));
    }

    #[test]
    fn rejects_bad_token_counts_without_indexing_past_the_line() {
        let error = spice2cir_text("R1 VIN VOUT\n").unwrap_err();

        assert!(matches!(
            error,
            SpiceError::BadTokenCount {
                line: 1,
                expected: 4,
                actual: 3,
                ..
            }
        ));
    }

    #[test]
    fn rejects_elements_outside_the_supported_linear_subset() {
        let error = spice2cir_text("M1 D G S B model\n").unwrap_err();

        assert!(matches!(
            error,
            SpiceError::UnsupportedElement { line: 1, .. }
        ));
    }

    #[test]
    fn filesystem_wrapper_writes_the_in_memory_conversion() {
        let root =
            std::env::temp_dir().join(format!("shapeic-spice2cir-test-{}", std::process::id()));
        let input = root.join("input");
        let output = root.join("output");
        fs::create_dir_all(&input).unwrap();
        fs::write(
            input.join("divider.spice"),
            "V1 VIN 0 1\nR1 VIN VOUT resistance\nR2 VOUT VSS resistance\n.end\n",
        )
        .unwrap();

        let nodes = spice2cir(&input, &output, "divider").unwrap();
        let converted = fs::read_to_string(output.join("divider.cir")).unwrap();

        assert_eq!(
            converted,
            "V1 1 0 1\nR1 1 2 resistance\nR2 2 0 resistance\n"
        );
        assert_eq!(nodes.get("VIN"), Some(1));
        assert_eq!(nodes.get("VOUT"), Some(2));
        fs::remove_dir_all(root).unwrap();
    }
}
