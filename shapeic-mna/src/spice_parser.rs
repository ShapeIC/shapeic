use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::path::Path;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone)]
pub struct NodeMap {
    pub nodes: HashMap<String, usize>,
}

#[derive(Debug)]
pub enum SpiceError {
    Io(std::io::Error),
    EmptyLine,
}

impl From<std::io::Error> for SpiceError {
    fn from(error: std::io::Error) -> Self {
        SpiceError::Io(error)
    }
}

pub fn spice_parser(
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

        if line.trim().is_empty() {
            continue;
        }

        let first_char = line.chars().next().ok_or(SpiceError::EmptyLine)?;

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
