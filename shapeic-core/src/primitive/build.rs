use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::primitive::manifest::PrimitiveManifest;
use shapeic_lut::DeviceLut;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveBuildSpec {
    #[serde(default)]
    pub inputs: Vec<PrimitiveBuildInputSpec>,
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

pub struct PrimitiveBuildEngine<B> {
    lut_backend: B,
}
impl<B: LutBackend> PrimitiveBuildEngine<B> {
    pub fn new(lut_backend: B) -> Self {
        Self { lut_backend }
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
        })?;
    
    build(model, build_spec, &input);
}

fn build(
    model: &DeviceLut,
    build_spec: &PrimitiveBuildSpec,
    input: &PrimitiveBuildInput,
) {
    let mut rows = expand_inputs(build_spec, input)?;
    evaluate_expressions(&mut rows, &build_spec.derived)?;
    model.size_for_current(); 
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
            let value = ExpressionParser::new(&expression.expr, row).parser().map_err(|reason| {
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
