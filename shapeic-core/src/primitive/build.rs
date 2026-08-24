use std::collections::HashMap;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::primitive::manifest::PrimitiveManifest;
use shapeic_lut::{
    CurrentSizingResult, DeviceLut, Expr, MosCapacitanceMatrix, MosExpression, OperatingPoint,
};
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
    LutSizing {
        lut: String,
        reason: String,
    },
    MissingLutExtrinsicCapacitances {
        lut: String,
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

    fn into_candidate_set(
        self,
        instance_name: &str,
    ) -> Result<CandidateSet, CandidateSetBuildError> {
        let columns = self
            .columns
            .into_iter()
            .map(|column| ExplorationColumn {
                name: primitive_build_candidate_column_name(instance_name, &column.name),
                values: column.values,
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
    pub current: String,
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
        })?;

    let mut input = input;
    if input.lut_config.is_none() {
        input.lut_config = primitive.lut_config.clone();
    }
    build(model, build_spec, &input)?
        .into_candidate_set(instance_name)
        .map_err(PrimitiveBuildError::CandidateSet)
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
    columns.extend(exposed_lut_provenance_columns(build_spec, &rows)?);
    columns.extend(exposed_input_columns(build_spec, &rows)?);

    Ok(PrimitiveBuildOutput {
        row_count: rows.len(),
        columns,
    })
}

fn exposed_lut_provenance_columns(
    spec: &PrimitiveBuildSpec,
    rows: &[HashMap<String, f64>],
) -> Result<Vec<ExplorationColumn>, PrimitiveBuildError> {
    let existing_columns = spec
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect::<HashSet<_>>();
    let mut exposed = Vec::new();

    for lut in &spec.lut {
        for parameter in ["length", "finger_width", "nf", "vbs", "vgs", "vds"] {
            let column_name = format!("{parameter}__{}", lut.name);
            if existing_columns.contains(column_name.as_str()) {
                continue;
            }
            let symbol = format!("lut.{}.{parameter}", lut.name);
            let values = rows
                .iter()
                .map(|row| {
                    row.get(&symbol)
                        .copied()
                        .ok_or_else(|| PrimitiveBuildError::MissingSymbol {
                            symbol: symbol.clone(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            exposed.push(ExplorationColumn::new(column_name, values));
        }
    }

    Ok(exposed)
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
            let requested_current = evaluate_lut_current(lut, row)?;

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
                    current: requested_current,
                });
                query_rows.push(row.clone());
            }
        }
        let expressions = lut_expressions(model, lut)?;
        let lut_results = many_size_for_current(model, &queries, &expressions)?;
        let mut next_rows = Vec::with_capacity(query_rows.len());
        for (row, lut_values) in query_rows.into_iter().zip(lut_results) {
            let mut next_row = row;
            insert_lut_result(&mut next_row, lut, &expressions, lut_values)?;
            next_rows.push(next_row);
        }
        *rows = next_rows;
    }
    Ok(())
}

fn lut_expressions(
    model: &DeviceLut,
    lut: &LutBuildSpec,
) -> Result<Vec<Expr>, PrimitiveBuildError> {
    let standard = |kind| {
        model
            .standard_expression(kind)
            .map_err(|error| PrimitiveBuildError::LutSizing {
                lut: lut.name.clone(),
                reason: error.to_string(),
            })
    };
    let mut expressions = vec![
        standard(MosExpression::Gmid)?,
        standard(MosExpression::CurrentDensity)?,
        standard(MosExpression::InverseEarlyVoltage)?,
    ];
    expressions.extend(
        MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
            .into_iter()
            .map(Expr::parameter),
    );
    Ok(expressions)
}

fn insert_lut_result(
    row: &mut HashMap<String, f64>,
    lut: &LutBuildSpec,
    expressions: &[Expr],
    sizing: CurrentSizingResult,
) -> Result<(), PrimitiveBuildError> {
    let extrinsic = sizing.extrinsic_capacitances.ok_or_else(|| {
        PrimitiveBuildError::MissingLutExtrinsicCapacitances {
            lut: lut.name.clone(),
        }
    })?;
    let prefix = format!("lut.{}", lut.name);
    row.insert(
        format!("{prefix}.length"),
        sizing.point.operating_point.length,
    );
    row.insert(format!("{prefix}.nf"), f64::from(sizing.nf));
    row.insert(
        format!("{prefix}.finger_width"),
        sizing.point.finger_width,
    );
    row.insert(format!("{prefix}.total_width"), sizing.total_width);
    row.insert(
        format!("{prefix}.vbs"),
        sizing.point.operating_point.vbs,
    );
    row.insert(
        format!("{prefix}.vgs"),
        sizing.point.operating_point.vgs,
    );
    row.insert(
        format!("{prefix}.vds"),
        sizing.point.operating_point.vds,
    );
    for (key, value) in expressions.iter().zip(sizing.values) {
        row.insert(
            format!("{prefix}.{}", key.parameter_name().unwrap()),
            value,
        );
    }
    row.insert(format!("{prefix}.cgsol_total"), extrinsic.cgsol);
    row.insert(format!("{prefix}.cgdol_total"), extrinsic.cgdol);
    row.insert(format!("{prefix}.cjs_total"), extrinsic.cjs);
    row.insert(format!("{prefix}.cjd_total"), extrinsic.cjd);
    Ok(())
}

fn evaluate_lut_current(
    lut: &LutBuildSpec,
    row: &HashMap<String, f64>,
) -> Result<f64, PrimitiveBuildError> {
    ExpressionParser::new(&lut.current, row)
        .parse()
        .map_err(|reason| PrimitiveBuildError::Expression {
            name: format!("lut.{}.current", lut.name),
            expression: lut.current.clone(),
            reason,
        })
}

fn many_size_for_current(
    model: &DeviceLut,
    queries: &[LutQuery],
    expressions: &[Expr],
) -> Result<Vec<CurrentSizingResult>, PrimitiveBuildError> {
    let mut lut_results = Vec::with_capacity(queries.len());
    for query in queries {
        let size = model
            .size_for_current(&query.op, query.current, expressions)
            .map_err(|error| PrimitiveBuildError::LutSizing {
                lut: query.lut_name.clone(),
                reason: error.to_string(),
            })?;
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use shapeic_lut::{LookupTable, LutPoint, MosExtrinsicCapacitances};

    use super::*;

    fn lut_spec(current: &str) -> LutBuildSpec {
        LutBuildSpec {
            name: "m1".to_owned(),
            device: "nmos".to_owned(),
            current: current.to_owned(),
            dof: HashMap::new(),
            lengths: Some(LutLengths::Values(vec![0.4e-6])),
        }
    }

    fn build_lut_expressions() -> Vec<Expr> {
        let mut expressions = vec![
            Expr::parameter("gmid"),
            Expr::parameter("jd"),
            Expr::parameter("gdsid"),
        ];
        expressions.extend(
            MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
                .into_iter()
                .map(Expr::parameter),
        );
        expressions
    }

    fn sizing_result(
        extrinsic_capacitances: Option<MosExtrinsicCapacitances>,
    ) -> CurrentSizingResult {
        let operating_point = OperatingPoint::new(0.4e-6, 0.0, 0.4, 0.4);
        CurrentSizingResult {
            point: LutPoint::new(operating_point, 0.75e-6),
            nf: 3,
            total_width: 2.25e-6,
            requested_current: 30.0e-6,
            finger_current: 10.0e-6,
            predicted_current: 30.0e-6,
            current_error: 0.0,
            values: vec![
                12.0, 20.0, 0.02, 1.0e-15, 2.0e-15, 3.0e-15, 4.0e-15, 5.0e-15,
                6.0e-15, 7.0e-15, 8.0e-15, 9.0e-15,
            ],
            extrinsic_capacitances,
        }
    }

    #[test]
    fn lut_current_uses_the_derived_branch_current() {
        let mut rows = vec![HashMap::from([("current".to_owned(), 20.0e-6)])];
        evaluate_expressions(
            &mut rows,
            &[BuildExpression {
                name: "id_m1".to_owned(),
                expr: "current / 2".to_owned(),
            }],
        )
        .expect("derived current should evaluate");

        let current = evaluate_lut_current(&lut_spec("id_m1"), &rows[0])
            .expect("LUT current should resolve");

        assert_eq!(current, 10.0e-6);
    }

    #[test]
    fn lut_current_supports_a_direct_primitive_current() {
        let mut rows = vec![HashMap::from([("current".to_owned(), 20.0e-6)])];
        evaluate_expressions(
            &mut rows,
            &[BuildExpression {
                name: "id_m1".to_owned(),
                expr: "current".to_owned(),
            }],
        )
        .expect("derived current should evaluate");

        let current = evaluate_lut_current(&lut_spec("id_m1"), &rows[0])
            .expect("LUT current should resolve");

        assert_eq!(current, 20.0e-6);
    }

    #[test]
    fn missing_lut_current_symbol_has_lut_context() {
        let error = evaluate_lut_current(&lut_spec("id_m1"), &HashMap::new())
            .expect_err("missing branch current should fail");

        assert_eq!(
            error,
            PrimitiveBuildError::Expression {
                name: "lut.m1.current".to_owned(),
                expression: "id_m1".to_owned(),
                reason: "missing symbol 'id_m1'".to_owned(),
            }
        );
    }

    #[test]
    fn non_positive_lut_current_returns_a_build_error() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../shapeic-lut/tests/fixtures/shapeic_v2_5d.npz");
        let table = LookupTable::open(fixture).expect("five-dimensional LUT fixture should load");
        let model = table.model("sg13_lv_nmos").expect("fixture nmos");
        let queries = [LutQuery {
            lut_name: "m1".to_owned(),
            device: "nmos".to_owned(),
            dof: HashMap::new(),
            length: 0.4e-6,
            op: OperatingPoint::new(0.4e-6, 0.0, 0.4, 0.4),
            current: 0.0,
        }];

        let error = many_size_for_current(model, &queries, &[])
            .expect_err("non-positive LUT current should fail");

        assert!(matches!(
            error,
            PrimitiveBuildError::LutSizing { lut, reason }
                if lut == "m1" && reason.contains("finite and positive")
        ));
    }

    #[test]
    fn analog_primitives_declare_their_lut_current() {
        for json in [
            include_str!("../../../analoglib/primitives/simplediffpair/build.json"),
            include_str!("../../../analoglib/primitives/simplecurrentmirror/build.json"),
        ] {
            let spec: PrimitiveBuildSpec =
                serde_json::from_str(json).expect("primitive build spec should deserialize");
            assert_eq!(spec.lut[0].current, "id_m1");
        }
    }

    #[test]
    fn current_mirror_uses_reference_gate_and_output_drain_biases() {
        let spec: PrimitiveBuildSpec = serde_json::from_str(include_str!(
            "../../../analoglib/primitives/simplecurrentmirror/build.json"
        ))
        .expect("current-mirror build spec should deserialize");
        let mut rows = vec![HashMap::from([
            ("current".to_owned(), 20.0e-6),
            ("VINP".to_owned(), 0.65),
            ("VOUTP".to_owned(), 0.95),
            ("VDD".to_owned(), 1.2),
        ])];

        evaluate_expressions(&mut rows, &spec.derived)
            .expect("current-mirror operating point should evaluate");

        assert_eq!(rows[0]["vgs_m1"], 0.65 - 1.2);
        assert_eq!(rows[0]["vds_m1"], 0.95 - 1.2);
    }

    #[test]
    fn lut_current_is_required_by_the_build_schema() {
        let error = serde_json::from_str::<LutBuildSpec>(
            r#"{"name":"m1","device":"nmos","lengths":[4e-7]}"#,
        )
        .expect_err("LUT current must be explicit");

        assert!(error.to_string().contains("missing field `current`"));
    }

    #[test]
    fn exposes_sizing_geometry_and_capacitances_to_build_expressions() {
        let extrinsic = MosExtrinsicCapacitances {
            cgsol: 10.0e-15,
            cgdol: 11.0e-15,
            cjs: 12.0e-15,
            cjd: 13.0e-15,
        };
        let mut row = HashMap::new();

        insert_lut_result(
            &mut row,
            &lut_spec("id_m1"),
            &build_lut_expressions(),
            sizing_result(Some(extrinsic)),
        )
        .expect("complete sizing data should be exposed");

        assert_eq!(row["lut.m1.nf"], 3.0);
        assert_eq!(row["lut.m1.finger_width"], 0.75e-6);
        assert_eq!(row["lut.m1.total_width"], 2.25e-6);
        assert_eq!(row["lut.m1.vbs"], 0.0);
        assert_eq!(row["lut.m1.vgs"], 0.4);
        assert_eq!(row["lut.m1.vds"], 0.4);
        assert_eq!(row["lut.m1.cgg"], 1.0e-15);
        assert_eq!(row["lut.m1.css"], 9.0e-15);
        assert_eq!(row["lut.m1.cgsol_total"], extrinsic.cgsol);
        assert_eq!(row["lut.m1.cgdol_total"], extrinsic.cgdol);
        assert_eq!(row["lut.m1.cjs_total"], extrinsic.cjs);
        assert_eq!(row["lut.m1.cjd_total"], extrinsic.cjd);
    }

    #[test]
    fn rejects_sizing_results_without_extrinsic_capacitances() {
        let error = insert_lut_result(
            &mut HashMap::new(),
            &lut_spec("id_m1"),
            &build_lut_expressions(),
            sizing_result(None),
        )
        .expect_err("extrinsic capacitances are required by the build flow");

        assert_eq!(
            error,
            PrimitiveBuildError::MissingLutExtrinsicCapacitances {
                lut: "m1".to_owned(),
            }
        );
    }

    #[test]
    fn reports_missing_intrinsic_capacitance_parameters_with_lut_context() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../shapeic-lut/tests/fixtures/shapeic_v2_5d.npz");
        let table = LookupTable::open(fixture).expect("five-dimensional LUT fixture should load");
        let model = table.model("sg13_lv_nmos").expect("fixture nmos");
        let lut = lut_spec("id_m1");
        let operating_point = OperatingPoint::new(0.4e-6, 0.0, 0.4, 0.4);
        let finger_width = model.finger_widths().expect("finger-width axis")[0];
        let current = model
            .query_parameter_at(&LutPoint::new(operating_point, finger_width), "id")
            .expect("fixture drain current");
        let queries = [LutQuery {
            lut_name: lut.name.clone(),
            device: lut.device.clone(),
            dof: HashMap::new(),
            length: 0.4e-6,
            op: operating_point,
            current,
        }];
        let expressions = lut_expressions(model, &lut).expect("standard expressions");

        let error = many_size_for_current(model, &queries, &expressions)
            .expect_err("fixture intentionally lacks the complete intrinsic set");

        assert!(matches!(
            error,
            PrimitiveBuildError::LutSizing { lut, reason }
                if lut == "m1" && reason.contains("cdg")
        ));
    }

    #[test]
    fn analog_builds_publish_total_capacitances_for_both_branches() {
        for json in [
            include_str!("../../../analoglib/primitives/simplediffpair/build.json"),
            include_str!("../../../analoglib/primitives/simplecurrentmirror/build.json"),
        ] {
            let spec: PrimitiveBuildSpec =
                serde_json::from_str(json).expect("primitive build spec should deserialize");
            let mut row = HashMap::from([
                ("id_m1".to_owned(), 10.0e-6),
                ("lut.m1.length".to_owned(), 0.4e-6),
                ("lut.m1.total_width".to_owned(), 2.25e-6),
                ("lut.m1.gmid".to_owned(), 12.0),
                ("lut.m1.gdsid".to_owned(), 0.02),
                ("lut.m1.nf".to_owned(), 3.0),
                ("lut.m1.cgsol_total".to_owned(), 10.0e-15),
                ("lut.m1.cgdol_total".to_owned(), 11.0e-15),
                ("lut.m1.cjs_total".to_owned(), 12.0e-15),
                ("lut.m1.cjd_total".to_owned(), 13.0e-15),
            ]);
            for (index, parameter) in MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
                .into_iter()
                .enumerate()
            {
                row.insert(
                    format!("lut.m1.{parameter}"),
                    (index as f64 + 1.0) * 1.0e-15,
                );
            }

            evaluate_expressions(std::slice::from_mut(&mut row), &spec.columns)
                .expect("capacitance columns should evaluate");

            assert_eq!(row["width__m1"], 2.25e-6);
            for (index, parameter) in MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
                .into_iter()
                .enumerate()
            {
                let expected = (index as f64 + 1.0) * 1.0e-15 * 3.0;
                assert_eq!(row[&format!("{parameter}__m1")], expected);
                assert_eq!(row[&format!("{parameter}__m2")], expected);
            }
            for (parameter, expected) in [
                ("cgsol", 10.0e-15),
                ("cgdol", 11.0e-15),
                ("cjs", 12.0e-15),
                ("cjd", 13.0e-15),
            ] {
                assert_eq!(row[&format!("{parameter}__m1")], expected);
                assert_eq!(row[&format!("{parameter}__m2")], expected);
            }
        }
    }

    #[test]
    fn exposes_each_lut_query_coordinate_as_candidate_provenance() {
        let spec: PrimitiveBuildSpec = serde_json::from_str(include_str!(
            "../../../analoglib/primitives/simplediffpair/build.json"
        ))
        .expect("diff-pair build spec should deserialize");
        let rows = [HashMap::from([
            ("lut.m1.length".to_owned(), 0.8e-6),
            ("lut.m1.finger_width".to_owned(), 1.25e-6),
            ("lut.m1.nf".to_owned(), 4.0),
            ("lut.m1.vbs".to_owned(), 0.0),
            ("lut.m1.vgs".to_owned(), 0.25),
            ("lut.m1.vds".to_owned(), 0.35),
        ])];

        let columns = exposed_lut_provenance_columns(&spec, &rows)
            .expect("complete LUT provenance should be exposed");
        assert_eq!(
            columns
                .iter()
                .map(|column| column.name.as_str())
                .collect::<Vec<_>>(),
            ["finger_width__m1", "vbs__m1", "vgs__m1", "vds__m1"]
        );
        assert_eq!(columns[0].values, [1.25e-6]);
        assert_eq!(columns[1].values, [0.0]);
        assert_eq!(columns[2].values, [0.25]);
        assert_eq!(columns[3].values, [0.35]);
    }

    #[test]
    fn capacitance_columns_map_to_branch_specific_candidate_names() {
        assert_eq!(
            primitive_build_candidate_column_name("xdp", "cgg__m1"),
            "cgg__xdp__m1"
        );
        assert_eq!(
            primitive_build_candidate_column_name("xdp", "cgg__m2"),
            "cgg__xdp__m2"
        );
        assert_eq!(
            primitive_build_candidate_column_name("xcm", "cgsol__m1"),
            "cgsol__xcm__m1"
        );
        assert_eq!(
            primitive_build_candidate_column_name("xcm", "cgsol__m2"),
            "cgsol__xcm__m2"
        );
        for parameter in ["length", "finger_width", "nf", "vbs", "vgs", "vds"] {
            assert_eq!(
                primitive_build_candidate_column_name("xdp", &format!("{parameter}__m1")),
                format!("{parameter}__xdp__m1")
            );
        }
    }
}
