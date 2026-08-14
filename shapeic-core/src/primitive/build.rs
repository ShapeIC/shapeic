use std::collections::HashMap;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::primitive::manifest::PrimitiveManifest;
use shapeic_lut::{CurrentSizingResult, DeviceLut, Expr, OperatingPoint, MosExpression};
use crate::exploration::candidate::{CandidateSet, CandidateSetBuildError, candidate_set_from_columns, candidate_column_name};
use crate::exploration::table::ExplorationColumn;
use crate::netlist::names::small_signal_param_name;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveBuildSpec {
    #[serde(default)]
    pub inputs: Vec<PrimitiveBuildInputSpec>,
    #[serde(default)]
    pub sweep_mode: SweepMode,
    #[serde(default)]
    pub derived: Vec<BuildExpression>,
    #[serde(default)]
    pub lut: Vec<LutBuildSpec>,
    #[serde(default)]
    pub columns: Vec<BuildExpression>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimitiveBuildError {
    MissingBuildSpec {
        primitive: String,
    },
    AlignedLengthMismatch {
        input: String,
        expected: usize,
        actual: usize,
    },
    Expression {
        name: String,
        expression: String,
        reason: String,
    },
    MissingLutLengths {
        lut: String,
    },
    InvalidLutLengthsRef {
        lut: String,
        reference: String,
    },
    MissingSymbol {
        symbol: String,
    },
    CandidateSet(CandidateSetBuildError),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrimitiveBuildInputSpec {
    pub name: String,
    pub kind: PrimitiveBuildInputKind,
    #[serde(default)]
    pub required: bool,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveBuildInputKind {
    Scalar,
    Vector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SweepMode {
    #[default]
    Aligned,
    Cartesian,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveBuildOutput {
    pub columns: Vec<ExplorationColumn>,
    pub row_count: usize,
}

impl PrimitiveBuildOutput {
    pub fn to_candidate_set(
        &self,
        instance_name: &str,
    ) -> Result<CandidateSet, CandidateSetBuildError> {
        let columns = self
            .columns
            .iter()
            .map(|column| ExplorationColumn {
                name: primitive_build_candidate_column_name(instance_name, &column.name),
                values: column.values.clone(),
            })
            .collect::<Vec<_>>();
        candidate_set_from_columns(instance_name, &columns)
    }
}

fn primitive_build_candidate_column_name(instance_name: &str, column_name: &str) -> String {
    if let Some((param, branch)) = column_name.split_once("__") {
        if !param.is_empty() && !branch.is_empty() && !branch.contains("__") {
            return small_signal_param_name(param, instance_name, branch);
        }
    }

    candidate_column_name(instance_name, column_name)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildExpression {
    pub name: String,
    pub expr: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LutBuildSpec {
    pub name: String,
    pub device: String,
    #[serde(default)]
    pub dof: HashMap<String, String>,
    pub lengths: Option<LutLengths>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LutLengths {
    Values(Vec<f64>),
    Ref(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveBuildValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl PrimitiveBuildValue {
    fn values(&self) -> &[f64] {
        match self {
            Self::Scalar(value) => std::slice::from_ref(value),
            Self::Vector(values) => values,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PrimitiveBuildInput {
    pub values: HashMap<String, PrimitiveBuildValue>,
    pub lut_config: Option<serde_json::Value>,
}

impl PrimitiveBuildInput {
    pub fn new(values: HashMap<String, PrimitiveBuildValue>) -> Self {
        Self {
            values,
            lut_config: None,
        }
    }
}

pub fn build_candidate_set_for_primitive(
    model: &DeviceLut,
    primitive: &PrimitiveManifest,
    instance_name: &str,
    input: PrimitiveBuildInput
) -> Result<CandidateSet, PrimitiveBuildError> {
    let build_spec = primitive.build.as_ref()
        .ok_or_else(|| PrimitiveBuildError::MissingBuildSpec {
            primitive: primitive.name.clone(),
        }).unwrap();

    let mut input = input.clone();
    if input.lut_config.is_none() {
        input.lut_config = primitive.lut_config.clone();
    }
    build(model, build_spec, &input)?.to_candidate_set(instance_name).map_err(PrimitiveBuildError::CandidateSet)
}

fn build(
    model: &DeviceLut,
    build_spec: &PrimitiveBuildSpec,
    input: &PrimitiveBuildInput,
) -> Result<PrimitiveBuildOutput, PrimitiveBuildError>{
    let mut rows = expand_inputs(build_spec, input)?;
    evaluate_expressions(&mut rows, &build_spec.derived)?;
    lut_query(model, build_spec, &mut rows, input)?;
    evaluate_expressions(&mut rows, &build_spec.columns)?;

    let columns = build_spec
        .columns
        .iter()
        .map(|column| {
            let values = rows
            .iter()
            .map(|row| {
                row.get(&column.name).copied().ok_or_else(|| {
                    PrimitiveBuildError::MissingSymbol {
                        symbol: column.name.clone(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ExplorationColumn {
            name: column.name.clone(),
            values,
        })
    })
    .collect::<Result<Vec<_>, _>>()?;
    let mut columns = columns;
    columns.extend(exposed_input_columns(build_spec, &rows)?);

    Ok(PrimitiveBuildOutput {
        row_count: rows.len(),
        columns,
    })
}

fn exposed_input_columns(
    spec: &PrimitiveBuildSpec,
    rows: &[HashMap<String, f64>],
) -> Result<Vec<ExplorationColumn>, PrimitiveBuildError> {
    let existing_columns = spec
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect::<HashSet<_>>();
    let mut exposed = Vec::new();

    for input in &spec.inputs {
        if input.source.as_deref() != Some("port_voltage") {
            continue;
        }

        let column_name = input.name.to_ascii_lowercase();
        if existing_columns.contains(column_name.as_str()) {
            continue;
        }

        let values = rows
            .iter()
            .map(|row| {
                row.get(&input.name)
                    .copied()
                    .ok_or_else(|| PrimitiveBuildError::MissingSymbol {
                        symbol: input.name.clone(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        exposed.push(ExplorationColumn::new(column_name, values));
    }

    Ok(exposed)
}

#[derive(Debug, Clone, PartialEq)]
pub struct LutQuery {
    pub lut_name: String,
    pub device: String,
    pub dof: HashMap<String, f64>,
    pub length: f64,
    pub op: OperatingPoint,
    pub current: f64
}

fn lut_query(
    model: &DeviceLut,
    spec: &PrimitiveBuildSpec,
    rows: &mut Vec<HashMap<String, f64>>,
    input: &PrimitiveBuildInput
) -> Result<(), PrimitiveBuildError>{
    for lut in &spec.lut {
        let lengths = resolve_lut_lengths(lut, input)?;
        let mut query_rows = Vec::with_capacity(rows.len() * lengths.len());
        let mut queries = Vec::with_capacity(rows.len() * lengths.len());

        for row in rows.iter() {
            for length in &lengths {
                let mut dof = HashMap::new();
                for (dof_name, expr) in &lut.dof {
                    let value = ExpressionParser::new(expr, row).parse().map_err(|reason| {
                        PrimitiveBuildError::Expression {
                            name: format!("lut.{}.{}", lut.name, dof_name),
                            expression: expr.clone(),
                            reason,
                        }
                    })?;
                    dof.insert(dof_name.clone(), value);
                }

                let requested_current = row
                    .get("current")
                    .copied()
                    .ok_or_else(|| PrimitiveBuildError::Expression {
                        name: "current".to_string(),
                        expression: "current".to_string(),
                        reason: "missing input 'current'".to_string(),
                    })?;

                let vbs = match dof.get("vbs") {
                    Some(value) => *value,
                    None => 0.0,
                };
                let vgs = match dof.get("vgs") {
                    Some(value) => *value,
                    None => 0.0,
                };
                let vds = match dof.get("vds") {
                    Some(value) => *value,
                    None => 0.0,
                };

                let operating_point = OperatingPoint::new(*length, vbs, vgs, vds);

                queries.push(LutQuery {
                    lut_name: lut.name.clone(),
                    device: lut.device.clone(),
                    dof,
                    length: *length,
                    op: operating_point,
                    current: requested_current
                });
                query_rows.push((row.clone(), *length));
            }
        }
        let gmid = model.standard_expression(MosExpression::Gmid).expect("gmid expression");
        let jd = model.standard_expression(MosExpression::CurrentDensity).expect("jd expression"); 
        let expressions = &vec![gmid, jd, Expr::parameter("gds"), Expr::parameter("id")];
        let lut_results = many_size_for_current(model, &queries, &expressions).unwrap();
        //println!("lut_results: {:?}", lut_results);
        let mut next_rows = Vec::with_capacity(query_rows.len());
        for ((row, length), lut_values) in query_rows.into_iter().zip(lut_results) {
            let mut next_row = row;
            next_row.insert(format!("lut.{}.length", lut.name), length);
            for (key, value) in expressions.iter().zip(lut_values.values) {
                next_row.insert(format!("lut.{}.{}", lut.name, key.parameter_name().unwrap()), value);
            }
            next_rows.push(next_row);
        }
        *rows = next_rows;
    }
    Ok(())
}

fn many_size_for_current(
    model: &DeviceLut,
    queries: &Vec<LutQuery>,
    expressions: &Vec<Expr>
) -> Result<Vec<CurrentSizingResult>, PrimitiveBuildError> {
    let mut lut_results = Vec::with_capacity(queries.len());
    for query in queries {
        let size = model.size_for_current(
            &query.op, 
            query.current,
            expressions
        ).unwrap();
        lut_results.push(size);
    }
    Ok(lut_results)
}

fn resolve_lut_lengths(
    lut: &LutBuildSpec,
    input: &PrimitiveBuildInput,
) -> Result<Vec<f64>, PrimitiveBuildError> {
    match &lut.lengths {
        Some(LutLengths::Values(values)) if !values.is_empty() => Ok(values.clone()),
        Some(LutLengths::Values(_)) => Err(PrimitiveBuildError::MissingLutLengths {
            lut: lut.name.clone(),
        }),
        Some(LutLengths::Ref(reference)) if reference == "$lut_config.lengths" => input
            .lut_config
            .as_ref()
            .and_then(|config| config.get("lengths"))
            .and_then(|lengths| lengths.as_array())
            .map(|lengths| {
                lengths
                    .iter()
                    .filter_map(serde_json::Value::as_f64)
                    .collect::<Vec<_>>()
            })
            .filter(|lengths| !lengths.is_empty())
            .ok_or_else(|| PrimitiveBuildError::InvalidLutLengthsRef {
                lut: lut.name.clone(),
                reference: reference.clone(),
            }),
        Some(LutLengths::Ref(reference)) => Err(PrimitiveBuildError::InvalidLutLengthsRef {
            lut: lut.name.clone(),
            reference: reference.clone(),
        }),
        None => Err(PrimitiveBuildError::MissingLutLengths {
            lut: lut.name.clone(),
        }),
    }
}

fn expand_inputs(
    spec: &PrimitiveBuildSpec,
    input: &PrimitiveBuildInput
) -> Result<Vec<HashMap<String, f64>>, PrimitiveBuildError>{
    let active_inputs = spec
        .inputs
        .iter()
        .filter_map(|input_spec| {
            input
                .values
                .get(&input_spec.name)
                .map(|value| (&input_spec.name, value))
        })
        .collect::<Vec<_>>();
    match spec.sweep_mode {
        SweepMode::Aligned => expand_aligned(active_inputs),
        SweepMode::Cartesian => expand_cartesian(active_inputs),
    }
}

fn expand_aligned(
    inputs: Vec<(&String, &PrimitiveBuildValue)>,
) -> Result<Vec<HashMap<String, f64>>, PrimitiveBuildError> {
    let row_count = inputs
        .iter()
        .filter_map(|(_, value)| match value {
            PrimitiveBuildValue::Vector(values) => Some(values.len()),
            PrimitiveBuildValue::Scalar(_) => None,
        })
        .max()
        .unwrap_or(1);

    for (name, value) in &inputs {
        if let PrimitiveBuildValue::Vector(values) = value {
            if values.len() != row_count {
                return Err(PrimitiveBuildError::AlignedLengthMismatch {
                    input: (*name).clone(),
                    expected: row_count,
                    actual: values.len(),
                });
            }
        }
    }

    let mut rows = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let mut row = HashMap::new();
        for (name, value) in &inputs {
            let value = match value {
                PrimitiveBuildValue::Scalar(value) => *value,
                PrimitiveBuildValue::Vector(values) => values[row_idx],
            };
            row.insert((*name).clone(), value);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn expand_cartesian(
    inputs: Vec<(&String, &PrimitiveBuildValue)>,
) -> Result<Vec<HashMap<String, f64>>, PrimitiveBuildError> {
    let mut rows = vec![HashMap::new()];

    for (name, value) in inputs {
        let mut next_rows = Vec::new();
        for row in &rows {
            for item in value.values() {
                let mut next_row = row.clone();
                next_row.insert(name.clone(), *item);
                next_rows.push(next_row);
            }
        }
        rows = next_rows;
    }

    Ok(rows)
}

fn evaluate_expressions(
    rows: &mut [HashMap<String, f64>],
    expressions: &[BuildExpression],
) -> Result<(), PrimitiveBuildError> {
    for expression in expressions {
        for row in rows.iter_mut() {
            let value = ExpressionParser::new(&expression.expr, row).parse().map_err(|reason| {
                PrimitiveBuildError::Expression {
                    name: expression.name.clone(),
                    expression: expression.expr.clone(),
                    reason,
                }
            })?;
            row.insert(expression.name.clone(), value);
        }
    }
    Ok(())
}

struct ExpressionParser<'a> {
    expression: &'a str,
    bytes: &'a [u8],
    pos: usize,
    symbols: &'a HashMap<String, f64>,
}

impl<'a> ExpressionParser<'a> {
    fn new(expression: &'a str, symbols: &'a HashMap<String, f64>) -> Self {
        Self {
            expression,
            bytes: expression.as_bytes(),
            pos: 0,
            symbols,
        }
    }

    fn parse(mut self) -> Result<f64, String> {
        let value = self.parse_expression()?;
        self.skip_whitespace();
        if self.pos != self.bytes.len() {
            return Err(format!("unexpected token at byte {}", self.pos));
        }
        Ok(value)
    }

    fn parse_expression(&mut self) -> Result<f64, String> {
        let mut value = self.parse_term()?;
        loop {
            self.skip_whitespace();
            if self.consume(b'+') {
                value += self.parse_term()?;
            } else if self.consume(b'-') {
                value -= self.parse_term()?;
            } else {
                break;
            }
        }
        Ok(value)
    }

    fn parse_term(&mut self) -> Result<f64, String> {
        let mut value = self.parse_factor()?;
        loop {
            self.skip_whitespace();
            if self.consume(b'*') {
                value *= self.parse_factor()?;
            } else if self.consume(b'/') {
                value /= self.parse_factor()?;
            } else {
                break;
            }
        }
        Ok(value)
    }

    fn parse_factor(&mut self) -> Result<f64, String> {
        self.skip_whitespace();
        if self.consume(b'-') {
            return Ok(-self.parse_factor()?);
        }
        if self.consume(b'+') {
            return self.parse_factor();
        }
        if self.consume(b'(') {
            let value = self.parse_expression()?;
            self.skip_whitespace();
            if !self.consume(b')') {
                return Err("missing closing ')'".to_string());
            }
            return Ok(value);
        }
        if self.peek().is_some_and(is_number_start) {
            return self.parse_number();
        }
        if self.peek().is_some_and(is_identifier_start) {
            return self.parse_symbol();
        }
        Err(format!("expected factor at byte {}", self.pos))
    }

    fn parse_number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.consume(b'.') {
            while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exponent_start = self.pos;
            while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                self.pos += 1;
            }
            if self.pos == exponent_start {
                return Err("missing exponent digits".to_string());
            }
        }
        self.expression[start..self.pos]
            .parse::<f64>()
            .map_err(|error| format!("invalid number: {error}"))
    }

    fn parse_symbol(&mut self) -> Result<f64, String> {
        let start = self.pos;
        while self.peek().is_some_and(is_identifier_body) {
            self.pos += 1;
        }
        let symbol = &self.expression[start..self.pos];
        self.symbols
            .get(symbol)
            .copied()
            .ok_or_else(|| format!("missing symbol '{symbol}'"))
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
}

fn is_number_start(byte: u8) -> bool {
    byte.is_ascii_digit() || byte == b'.'
}

fn is_identifier_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_identifier_body(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.')
}
