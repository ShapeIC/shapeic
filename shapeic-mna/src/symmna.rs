use symbolica::atom::Atom;
use symbolica::prelude::{Matrix, parse};
use symbolica::domains::atom::AtomField;

pub type Vector = Vec<Atom>;

#[derive(Debug)]
pub enum SmnaError {
    UnknownElement {
        line: usize,
        content: String,
    },
    BadTokenCount {
        line: usize,
        content: String,
        expected: usize,
        actual: usize,
    },
    BadNode {
        token: String,
    },
    BadValue {
        token: String,
    },
    MissingNode {
        node: usize,
    },
    MissingBranch {
        name: String,
    },
}

pub struct SmnaResult {
    pub report: String,
    pub a: Matrix<AtomField>,
    pub x: Vector,
    pub z: Matrix<AtomField>,
}

#[derive(Debug, Default)]
pub struct Counts {
    branch_cnt: usize,
    num_rlc: usize,
    num_ind: u32,
    num_v: u32,
    num_i: usize,
    num_opamps: u32,
    num_vcvs: u32,
    num_vccs: u32,
    num_cccs: u32,
    num_ccvs: u32,
    num_cpld_ind: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Branch {
    pub element: String,
    pub p_node: Option<usize>,
    pub n_node: Option<usize>,
    pub cp_node: Option<usize>,
    pub cn_node: Option<usize>,
    pub vout: Option<usize>,
    pub value: Option<Atom>,
    pub vname: Option<String>,
    pub lname1: Option<String>,
    pub lname2: Option<String>,
}

impl Branch {
    fn new(element: String) -> Self {
        Self {
            element,
            p_node: None,
            n_node: None,
            cp_node: None,
            cn_node: None,
            vout: None,
            value: None,
            vname: None,
            lname1: None,
            lname2: None,
        }
    }

    fn kind(&self) -> char {
        self.element.chars().next().unwrap_or('\0')
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurrentUnknown {
    pub element: String,
    pub p_node: Option<usize>,
    pub n_node: Option<usize>,
}

pub fn smna(net_list: &str) -> Result<SmnaResult, SmnaError> {
    let content = preprocess(net_list);
    let counts = validate_and_count_elements(&content)?;
    let line_cnt = content.len();
    
    let mut df = parse_branches(&content)?;
    move_voltage_sources_first(&mut df);
    let num_nodes = count_nodes(&df, line_cnt)?;

    let df2 = current_unknowns(&df);
    let i_unk = counts.num_v 
        + counts.num_opamps 
        + counts.num_vcvs
        + counts.num_cccs
        + counts.num_ind
        + counts.num_cccs;

    let mut g = Matrix::new(num_nodes,  num_nodes, AtomField::new());
    let mut b = Matrix::new(num_nodes, i_unk, AtomField::new());
    let mut c = Matrix::new(i_unk, num_nodes, AtomField::new());
    let mut d = Matrix::new(i_unk, i_unk, AtomField::new());
    let mut i_vec = Matrix::new(num_nodes, 1, AtomField::new());
    let mut ev = Matrix::new(i_unk, 1, AtomField::new());

    stamp_g(&df, &mut g);
    stamp_b(&df, &mut b, i_unk)?;
    stamp_c(&df, &df2, &mut c, i_unk)?;
    stamp_d(&df, &df2, &mut d, i_unk)?;
    stamp_i(&df, &mut i_vec);
    stamp_ev(&df, &mut ev);

    let mut x = Vec::with_capacity(usize::try_from(num_nodes + i_unk).expect("vector capacity does not fit in usize"));
    for node in 1..=num_nodes {
        x.push(parse!((format!("v{node}")).as_str()));
    }
    for unknown in &df2 {
        x.push(parse!((format!("I_{}", unknown.element)).as_str()));
    }

    let mut z = Matrix::new(num_nodes + i_unk, 1, AtomField::new());
    for row in 0..num_nodes {
        z[(row, 0)] = i_vec[(row, 0)].clone();
    }

    for row in 0..i_unk {
        z[(num_nodes + row, 0)] = ev[(row, 0)].clone();
    }

    let mut a = Matrix::new(num_nodes + i_unk, num_nodes + i_unk, AtomField::new());
    for row in 0..num_nodes {
        for col in 0..num_nodes {
            a[(row, col)] = g[(row, col)].clone();
        }
    }
    for row in 0..num_nodes {
        for col in 0..i_unk {
            a[(row, num_nodes + col)] = b[(row, col)].clone();
        }
    }
    for row in 0..i_unk {
        for col in 0..num_nodes {
            a[(num_nodes + row, col)] = c[(row, col)].clone();
        }
    }
    for row in 0..i_unk {
        for col in 0..i_unk {
            a[(num_nodes + row, num_nodes + col)] = d[(row, col)].clone();
        }
    }

    Ok(SmnaResult {
        report: report(&counts, line_cnt, num_nodes as usize, i_unk as usize),
        a,
        x,
        z,
    })
}

fn preprocess(net_list: &str) -> Vec<String> {
    net_list
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with('*'))
        .map(capitalize_first)
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

fn capitalize_first(line: &str) -> String {
    let mut chars = line.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn validate_and_count_elements(content: &[String]) -> Result<Counts, SmnaError> {
   let mut counts = Counts::default();

   for (line, item) in content.iter().enumerate() {
       let kind = item.chars().next().unwrap_or('\0');
       let actual = item.split_whitespace().count();
       match kind {
           'R' | 'L' | 'C' => {
               ensure_token_count(line, item, actual, 4)?;
               counts.num_rlc += 1;
               counts.branch_cnt += 1;
               if kind == 'L' {
                   counts.num_ind += 1;
               }
           }
           'V' => {
               ensure_token_count(line, item, actual, 4)?;
               counts.num_v += 1;
               counts.branch_cnt += 1;
           }
           'I' => {
               ensure_token_count(line, item, actual, 4)?;
               counts.num_i += 1;
               counts.branch_cnt += 1;
           }
           'O' => {
               ensure_token_count(line, item, actual, 4)?;
               counts.num_opamps += 1;
           }
           'E' => {
               ensure_token_count(line, item, actual, 6)?;
               counts.num_vcvs += 1;
               counts.branch_cnt += 1;
           }
           'G' => {
               ensure_token_count(line, item, actual, 6)?;
               counts.num_vccs += 1;
               counts.branch_cnt += 1;
           }
           'F' => {
               ensure_token_count(line, item, actual, 5)?;
               counts.num_cccs += 1;
               counts.branch_cnt += 1;
           }
           'H' => {
               ensure_token_count(line, item, actual, 5)?;
               counts.num_ccvs += 1;
               counts.branch_cnt += 1;
           }
           'K' => {
               ensure_token_count(line, item, actual, 4)?;
               counts.num_cpld_ind += 1;
           }
           _ => {
               return Err(SmnaError::UnknownElement {
                   line,
                   content: item.clone(),
               });
           }
       }
   }
   Ok(counts)
}

fn ensure_token_count(
    line: usize,
    content: &str,
    actual: usize,
    expected: usize,
) -> Result<(), SmnaError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SmnaError::BadTokenCount {
            line,
            content: content.to_string(),
            expected,
            actual,
        })
    }
}

fn parse_branches(content: &[String]) -> Result<Vec<Branch>, SmnaError> {
    let mut branches = Vec::with_capacity(content.len());

    for item in content {
        let tokens = item.split_whitespace().collect::<Vec<_>>();
        let kind = tokens[0].chars().next().unwrap_or('\0');
        let mut branch = Branch::new(match kind {
            'E' => tokens[0].replace('E', "Ea"),
            _ => tokens[0].to_string(),
        });

        match kind {
            'R' | 'L' | 'C' | 'V' | 'I' => {
                branch.p_node = Some(parse_node(tokens[1])?);
                branch.n_node = Some(parse_node(tokens[2])?);
                branch.value = Some(parse_value(tokens[3])?);
            }
            'O' => {
                branch.p_node = Some(parse_node(tokens[1])?);
                branch.n_node = Some(parse_node(tokens[2])?);
                branch.vout = Some(parse_node(tokens[3])?);
            }
            'G' | 'E' => {
                branch.p_node = Some(parse_node(tokens[1])?);
                branch.n_node = Some(parse_node(tokens[2])?);
                branch.cp_node = Some(parse_node(tokens[3])?);
                branch.cn_node = Some(parse_node(tokens[4])?);
                branch.value = Some(parse_value(tokens[5])?);
            }
            'F' | 'H' => {
                branch.p_node = Some(parse_node(tokens[1])?);
                branch.n_node = Some(parse_node(tokens[2])?);
                branch.vname = Some(capitalize_first(tokens[3]));
                branch.value = Some(parse_value(tokens[4])?);
            }
            'K' => {
                branch.lname1 = Some(capitalize_first(tokens[1]));
                branch.lname2 = Some(capitalize_first(tokens[2]));
                branch.value = Some(parse_value(tokens[3])?);
            }
            _ => unreachable!(),
        }

        branches.push(branch);
    }

    Ok(branches)
}

fn parse_node(token: &str) -> Result<usize, SmnaError> {
    token.parse().map_err(|_| SmnaError::BadNode {
        token: token.to_string(),
    })
}

fn parse_value(token: &str) -> Result<Atom, SmnaError> {
    if token.is_empty() {
        Err(SmnaError::BadValue {
            token: token.to_string(),
        })
    } else {
        Ok(parse!(token))
    }
}

fn move_voltage_sources_first(df: &mut Vec<Branch>) {
    let mut source = Vec::new();
    let mut other = Vec::new();

    for branch in df.drain(..) {
        if branch.kind() == 'V' {
            source.push(branch);
        } else {
            other.push(branch);
        }
    }

    source.extend(other);
    *df = source;
}

fn count_nodes(df: &[Branch], line_cnt: usize) -> Result<u32, SmnaError> {
    let mut present = vec![false; line_cnt + 1];
    let mut largest = 0;

    for branch in df {
        if branch.kind() == 'K' {
            continue;
        }
        for node in [
            branch.p_node,
            branch.n_node,
            branch.cp_node,
            branch.cn_node,
            branch.vout,
        ]
        .into_iter()
        .flatten()
        {
            if node < present.len() {
                present[node] = true;
            }
            largest = largest.max(node);
        }
    }

    for node in 1..largest {
        if !present.get(node).copied().unwrap_or(false) {
            return Err(SmnaError::MissingNode { node });
        }
    }

    Ok(matrix_index(largest))
}

fn current_unknowns(df: &[Branch]) -> Vec<CurrentUnknown> {
    df.iter()
        .filter(|branch| matches!(branch.kind(), 'L' | 'V' | 'O' | 'E' | 'H' | 'F'))
        .map(|branch| CurrentUnknown {
            element: branch.element.clone(),
            p_node: branch.p_node,
            n_node: branch.n_node,
        })
        .collect()
}
fn branch_symbol(branch: &Branch, lowercase: bool) -> Atom {
    branch.value.clone().unwrap_or_else(|| {
        let name = if lowercase {
            branch.element.to_lowercase()
        } else {
            branch.element.clone()
        };

        parse!(name.as_str())
    })
}

fn idx(node: usize) -> u32 {
    u32::try_from(node - 1)
        .expect("node index does not fit in u32")
}

fn stamp_g(df: &[Branch], g_matrix: &mut Matrix<AtomField>) {
    let s = parse!("s");
    let one = parse!("1");

    for branch in df {
        let kind = branch.kind();

        let n1 = branch.p_node.unwrap_or(0);
        let n2 = branch.n_node.unwrap_or(0);
        let cn1 = branch.cp_node.unwrap_or(0);
        let cn2 = branch.cn_node.unwrap_or(0);

        let g: Atom = match kind {
            'R' => {
                let r = branch_symbol(branch, false);
                &one / &r
            }
            'C' => {
                let c = branch_symbol(branch, false);
                &s * &c
            }
            'G' => branch_symbol(branch, true),
            _ => continue,
        };

        if matches!(kind, 'R' | 'C') {
            if n1 != 0 && n2 != 0 {
                g_matrix[(idx(n1), idx(n2))] -= &g;
                g_matrix[(idx(n2), idx(n1))] -= &g;
            }
            if n1 != 0 {
                g_matrix[(idx(n1), idx(n1))] += &g;
            }
            if n2 != 0 {
                g_matrix[(idx(n2), idx(n2))] += &g;
            }
        }

        if kind == 'G' {
            if n1 != 0 && cn1 != 0 {
                g_matrix[(idx(n1), idx(cn1))] += &g;
            }
            if n2 != 0 && cn2 != 0 {
                g_matrix[(idx(n2), idx(cn2))] += &g;
            }
            if n1 != 0 && cn2 != 0 {
                g_matrix[(idx(n1), idx(cn2))] -= &g;
            }
            if n2 != 0 && cn1 != 0 {
                g_matrix[(idx(n2), idx(cn1))] -= &g;
            }
        }
    }
}

fn stamp_b(df: &[Branch], b: &mut Matrix<AtomField>, i_unk: u32) -> Result<(), SmnaError> {
    let mut sn = 0;

    for branch in df {
        match branch.kind() {
            'V' | 'H' | 'F' | 'E' | 'L' => {
                stamp_current_column(
                    b,
                    sn,
                    branch.p_node.unwrap_or(0),
                    branch.n_node.unwrap_or(0),
                );
                sn += 1;
            }
            'O' => {
                if let Some(vout) = branch.vout.filter(|&vout| vout != 0) {
                    b[(idx(vout), sn as u32)] = parse!("1");
                }
                sn += 1;
            }
            _ => {}
        }
    }

    check_source_count("B", sn, i_unk)
}

fn stamp_current_column(matrix: &mut Matrix<AtomField>, col: usize, n1: usize, n2: usize) {
    if n1 != 0 {
        matrix[(idx(n1), col as u32)] = parse!("1");
    }
    if n2 != 0 {
        matrix[(idx(n2), col as u32)] = parse!("-1");
    }
}

fn check_source_count(matrix_name: &str, sn: usize, i_unk: u32) -> Result<(), SmnaError> {
    if sn as u32 == i_unk {
        Ok(())
    } else {
        Err(SmnaError::MissingBranch {
            name: format!("source count in matrix {matrix_name}: sn={sn}, i_unk={i_unk}"),
        })
    }
}

fn stamp_c(
    df: &[Branch],
    df2: &[CurrentUnknown],
    c: &mut Matrix<AtomField>,
    i_unk: u32,
) -> Result<(), SmnaError> {
    let mut sn = 0;

    for branch in df {
        match branch.kind() {
            'V' | 'O' | 'H' | 'L' => {
                stamp_current_row(
                    c,
                    sn,
                    branch.p_node.unwrap_or(0),
                    branch.n_node.unwrap_or(0),
                );
                sn += 1;
            }
            'F' => {
                sn += 1;
            }
            'E' => {
                stamp_current_row(
                    c,
                    sn,
                    branch.p_node.unwrap_or(0),
                    branch.n_node.unwrap_or(0),
                );
                let gain = branch_symbol(branch, true);
                if let Some(cn1) = branch.cp_node.filter(|&cn1| cn1 != 0) {
                    c[(sn as u32, idx(cn1))] = -&gain;
                }
                if let Some(cn2) = branch.cn_node.filter(|&cn2| cn2 != 0) {
                    c[(sn as u32, idx(cn2))] = gain;
                }
                sn += 1;
            }
            _ => {}
        }
    }

    let _ = df2;
    check_source_count("C", sn, i_unk)
}

fn stamp_current_row(matrix: &mut Matrix<AtomField>, row: usize, n1: usize, n2: usize) {
    if n1 != 0 {
        matrix[(row as u32, idx(n1))] = parse!("1");
    }
    if n2 != 0 {
        matrix[(row as u32, idx(n2))] = parse!("-1");
    }
}

fn stamp_d(
    df: &[Branch],
    df2: &[CurrentUnknown],
    d: &mut Matrix<AtomField>,
    i_unk: u32,
) -> Result<(), SmnaError> {
    let mut sn = 0;
    let s = parse!("s");

    for branch in df {
        match branch.kind() {
            'V' | 'O' | 'E' => sn += 1,
            'L' => {
                d[(sn as u32, sn as u32)] += &s*-branch_symbol(branch, false);
                sn += 1;
            }
            'H' => {
                let index = find_vname(df2, branch.vname.as_deref().unwrap_or(""))?;
                d[(sn as u32, index as u32)] += -branch_symbol(branch, true);
                sn += 1;
            }
            'F' => {
                let index = find_vname(df2, branch.vname.as_deref().unwrap_or(""))?;
                d[(sn as u32, index as u32)] += -branch_symbol(branch, true);
                d[(sn as u32, sn as u32)] = parse!("1");
                sn += 1;
            }
            'K' => {
                let ind1 = find_vname(df2, branch.lname1.as_deref().unwrap_or(""))?;
                let ind2 = find_vname(df2, branch.lname2.as_deref().unwrap_or(""))?;
                let suffix = branch
                    .element
                    .to_lowercase()
                    .trim_start_matches('k')
                    .to_string();
                let mutual = &s*-parse!((format!("M{suffix}")).as_str());
                d[(ind1 as u32, ind2 as u32)] += &mutual;
                d[(ind2 as u32, ind1 as u32)] += mutual;
            }
            _ => {}
        }
    }

    let _ = i_unk;
    Ok(())
}

fn find_vname(df2: &[CurrentUnknown], name: &str) -> Result<usize, SmnaError> {
    df2.iter()
        .position(|branch| branch.element == name)
        .ok_or_else(|| SmnaError::MissingBranch {
            name: name.to_string(),
        })
}

fn stamp_i(df: &[Branch], i_vec: &mut Matrix<AtomField>) {
    for branch in df {
        if branch.kind() != 'I' {
            continue;
        }

        let source = branch_symbol(branch, false);
        let n1 = branch.p_node.unwrap_or(0);
        let n2 = branch.n_node.unwrap_or(0);

        if n1 != 0 {
            i_vec[(idx(n1), 0)] -= &source;
        }
        if n2 != 0 {
            i_vec[(idx(n2), 0)] += &source;
        }
    }
}

fn stamp_ev(df: &[Branch], ev: &mut Matrix<AtomField>) {
    let mut sn = 0;
    for branch in df {
        if branch.kind() == 'V' {
            ev[(sn as u32, 0)] = branch_symbol(branch, false);
            sn += 1;
        }
    }
}

fn report(counts: &Counts, line_cnt: usize, num_nodes: usize, i_unk: usize) -> String {
    format!(
        "Net list report\n\
number of lines in netlist: {line_cnt}\n\
number of branches: {}\n\
number of nodes: {num_nodes}\n\
number of unknown currents: {i_unk}\n\
number of RLC (passive components): {}\n\
number of inductors: {}\n\
number of independent voltage sources: {}\n\
number of independent current sources: {}\n\
number of op amps: {}\n\
number of E - VCVS: {}\n\
number of G - VCCS: {}\n\
number of F - CCCS: {}\n\
number of H - CCVS: {}\n\
number of K - Coupled inductors: {}\n",
        counts.branch_cnt,
        counts.num_rlc,
        counts.num_ind,
        counts.num_v,
        counts.num_i,
        counts.num_opamps,
        counts.num_vcvs,
        counts.num_vccs,
        counts.num_cccs,
        counts.num_ccvs,
        counts.num_cpld_ind,
    )
}

fn matrix_index(index: usize) -> u32 {
    u32::try_from(index)
        .expect("MNA matrix index exceeds u32")
}


