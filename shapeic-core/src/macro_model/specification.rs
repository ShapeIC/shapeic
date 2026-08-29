//! Preparation and evaluation of analysis-independent macro specifications.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use crate::analysis::{AcMetric, AcMetrics};

use super::{
    Macro, MacroAcceptedCandidate, MacroCandidateSets, MacroSpecificationBounds,
    MacroSpecificationSource,
};

/// Prepared dependency order for one macro's specifications.
#[derive(Clone, Debug)]
pub(super) struct PreparedMacroSpecifications {
    evaluation_order: Vec<usize>,
    bounds: Vec<MacroSpecificationBounds>,
}

impl PreparedMacroSpecifications {
    pub(super) fn new(macro_: &Macro) -> Result<Self, MacroSpecificationEvaluationError> {
        let bounds = macro_
            .exploration()
            .specifications()
            .iter()
            .map(|specification| specification.bounds())
            .collect::<Vec<_>>();
        Self::new_with_bounds(macro_, bounds)
    }

    pub(super) fn new_with_bounds(
        macro_: &Macro,
        bounds: Vec<MacroSpecificationBounds>,
    ) -> Result<Self, MacroSpecificationEvaluationError> {
        let specifications = macro_.exploration().specifications();
        debug_assert_eq!(specifications.len(), bounds.len());
        let mut names = HashMap::with_capacity(specifications.len());
        for (index, specification) in specifications.iter().enumerate() {
            if specification.name().trim().is_empty() {
                return Err(MacroSpecificationEvaluationError::EmptyName { index });
            }
            if names.insert(specification.name(), index).is_some() {
                return Err(MacroSpecificationEvaluationError::DuplicateName {
                    name: specification.name().to_owned(),
                });
            }
            let effective_bounds = bounds[index];
            if effective_bounds
                .minimum()
                .is_some_and(|value| !value.is_finite())
                || effective_bounds
                    .maximum()
                    .is_some_and(|value| !value.is_finite())
                || effective_bounds
                    .minimum()
                    .zip(effective_bounds.maximum())
                    .is_some_and(|(minimum, maximum)| minimum > maximum)
            {
                return Err(MacroSpecificationEvaluationError::InvalidBounds {
                    specification: specification.name().to_owned(),
                });
            }
            validate_direct_source(macro_, specification.name(), specification.source())?;
        }

        let dependencies = specifications
            .iter()
            .map(|specification| match specification.source() {
                MacroSpecificationSource::Expression(expression) => {
                    Ok(expression_symbols(expression)?
                        .into_iter()
                        .filter_map(|symbol| names.get(symbol.as_str()).copied())
                        .collect::<Vec<_>>())
                }
                _ => Ok(Vec::new()),
            })
            .collect::<Result<Vec<_>, MacroSpecificationEvaluationError>>()?;
        let mut state = vec![VisitState::Unvisited; specifications.len()];
        let mut evaluation_order = Vec::with_capacity(specifications.len());
        for index in 0..specifications.len() {
            visit_specification(
                index,
                specifications,
                &dependencies,
                &mut state,
                &mut evaluation_order,
            )?;
        }
        Ok(Self {
            evaluation_order,
            bounds,
        })
    }

    pub(super) fn bounds(&self) -> &[MacroSpecificationBounds] {
        &self.bounds
    }

    pub(super) fn validate_context(
        &self,
        macro_: &Macro,
        candidate_sets: &MacroCandidateSets,
    ) -> Result<(), MacroSpecificationEvaluationError> {
        let specifications = macro_.exploration().specifications();
        let specification_names = specifications
            .iter()
            .map(|specification| specification.name())
            .collect::<HashSet<_>>();
        let mut external_symbols = HashSet::new();
        let complete_candidate_schema = candidate_sets
            .instances()
            .iter()
            .all(|instance| !instance.candidates().points.is_empty());

        for instance in candidate_sets.instances() {
            if let Some(point) = instance.candidates().points.first() {
                for (column, _) in &point.values {
                    if !external_symbols.insert(column.clone()) {
                        return Err(MacroSpecificationEvaluationError::DuplicateSymbol {
                            symbol: column.clone(),
                        });
                    }
                }
            }
        }
        for testbench in macro_.exploration().testbenches() {
            for (metric, name) in [
                (AcMetric::DcGainDb, "dc_gain_db"),
                (AcMetric::Bandwidth3DbHz, "bandwidth_3db_hz"),
                (AcMetric::UnityGainHz, "unity_gain_hz"),
                (AcMetric::PhaseMarginDeg, "phase_margin_deg"),
            ] {
                if testbench.analysis().policy.metrics.contains(metric) {
                    let symbol = format!("{}.{name}", testbench.name());
                    if !external_symbols.insert(symbol.clone()) {
                        return Err(MacroSpecificationEvaluationError::DuplicateSymbol { symbol });
                    }
                }
            }
        }
        if let Some(name) = specification_names
            .iter()
            .find(|name| external_symbols.contains::<str>(*name))
        {
            return Err(MacroSpecificationEvaluationError::DuplicateSymbol {
                symbol: (*name).to_owned(),
            });
        }

        for specification in specifications {
            match specification.source() {
                MacroSpecificationSource::CandidateColumn {
                    instance_path,
                    column,
                } => {
                    let instance = candidate_sets.instance(instance_path).ok_or_else(|| {
                        MacroSpecificationEvaluationError::UnknownCandidateInstance {
                            specification: specification.name().to_owned(),
                            instance_path: instance_path.clone(),
                        }
                    })?;
                    if instance
                        .candidates()
                        .points
                        .first()
                        .is_some_and(|point| point.get(column).is_none())
                    {
                        return Err(MacroSpecificationEvaluationError::MissingCandidateColumn {
                            specification: specification.name().to_owned(),
                            instance_path: instance_path.clone(),
                            column: column.clone(),
                        });
                    }
                }
                MacroSpecificationSource::Expression(expression) if complete_candidate_schema => {
                    for symbol in expression_symbols(expression)? {
                        if !specification_names.contains(symbol.as_str())
                            && !external_symbols.contains(symbol.as_str())
                        {
                            return Err(MacroSpecificationEvaluationError::MissingSymbol {
                                specification: specification.name().to_owned(),
                                symbol,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) fn evaluate(
        &self,
        macro_: &Macro,
        candidate_sets: &MacroCandidateSets,
        candidate: &MacroAcceptedCandidate,
    ) -> Result<Vec<f64>, MacroSpecificationEvaluationError> {
        let specifications = macro_.exploration().specifications();
        let mut symbols = candidate_symbols(macro_, candidate_sets, candidate)?;
        let mut values = vec![0.0; specifications.len()];
        for &index in &self.evaluation_order {
            let specification = &specifications[index];
            let value = match specification.source() {
                MacroSpecificationSource::CandidateColumn {
                    instance_path,
                    column,
                } => selected_candidate_value(candidate_sets, candidate, instance_path, column)
                    .ok_or_else(
                        || MacroSpecificationEvaluationError::MissingCandidateColumn {
                            specification: specification.name().to_owned(),
                            instance_path: instance_path.clone(),
                            column: column.clone(),
                        },
                    )?,
                MacroSpecificationSource::AcMetric { testbench, metric } => {
                    selected_metric(macro_, candidate, testbench, *metric).ok_or_else(|| {
                        MacroSpecificationEvaluationError::UnavailableMetric {
                            specification: specification.name().to_owned(),
                            testbench: testbench.clone(),
                            metric: *metric,
                        }
                    })?
                }
                MacroSpecificationSource::Expression(expression) => {
                    ExpressionParser::new(expression, &symbols)
                        .parse()
                        .map_err(|reason| MacroSpecificationEvaluationError::Expression {
                            specification: specification.name().to_owned(),
                            expression: expression.clone(),
                            reason,
                        })?
                }
            };
            if !value.is_finite() {
                return Err(MacroSpecificationEvaluationError::NonFiniteValue {
                    specification: specification.name().to_owned(),
                });
            }
            values[index] = value;
            if symbols
                .insert(specification.name().to_owned(), value)
                .is_some()
            {
                return Err(MacroSpecificationEvaluationError::DuplicateSymbol {
                    symbol: specification.name().to_owned(),
                });
            }
        }
        Ok(values)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VisitState {
    Unvisited,
    Visiting,
    Complete,
}

fn visit_specification(
    index: usize,
    specifications: &[super::MacroSpecification],
    dependencies: &[Vec<usize>],
    state: &mut [VisitState],
    order: &mut Vec<usize>,
) -> Result<(), MacroSpecificationEvaluationError> {
    match state[index] {
        VisitState::Complete => return Ok(()),
        VisitState::Visiting => {
            return Err(MacroSpecificationEvaluationError::DependencyCycle {
                specification: specifications[index].name().to_owned(),
            });
        }
        VisitState::Unvisited => {}
    }
    state[index] = VisitState::Visiting;
    for &dependency in &dependencies[index] {
        visit_specification(dependency, specifications, dependencies, state, order)?;
    }
    state[index] = VisitState::Complete;
    order.push(index);
    Ok(())
}

fn validate_direct_source(
    macro_: &Macro,
    specification: &str,
    source: &MacroSpecificationSource,
) -> Result<(), MacroSpecificationEvaluationError> {
    if let MacroSpecificationSource::AcMetric { testbench, metric } = source {
        let testbench_definition = macro_.exploration().testbench(testbench).ok_or_else(|| {
            MacroSpecificationEvaluationError::UnknownTestbench {
                specification: specification.to_owned(),
                testbench: testbench.clone(),
            }
        })?;
        if !testbench_definition
            .analysis()
            .policy
            .metrics
            .contains(*metric)
        {
            return Err(MacroSpecificationEvaluationError::MetricNotSelected {
                specification: specification.to_owned(),
                testbench: testbench.clone(),
                metric: *metric,
            });
        }
    }
    Ok(())
}

pub(super) fn accepted_candidate_symbols(
    macro_: &Macro,
    candidate_sets: &MacroCandidateSets,
    candidate: &MacroAcceptedCandidate,
) -> Result<HashMap<String, f64>, MacroSpecificationEvaluationError> {
    let mut symbols = candidate_symbols(macro_, candidate_sets, candidate)?;
    for (name, value) in candidate.specification_values() {
        if symbols.insert(name.clone(), *value).is_some() {
            return Err(MacroSpecificationEvaluationError::DuplicateSymbol {
                symbol: name.clone(),
            });
        }
    }
    Ok(symbols)
}

pub(super) fn evaluate_expression(
    expression: &str,
    symbols: &HashMap<String, f64>,
) -> Result<f64, String> {
    ExpressionParser::new(expression, symbols).parse()
}

fn candidate_symbols(
    macro_: &Macro,
    candidate_sets: &MacroCandidateSets,
    candidate: &MacroAcceptedCandidate,
) -> Result<HashMap<String, f64>, MacroSpecificationEvaluationError> {
    let mut symbols = HashMap::new();
    for (set_index, instance) in candidate_sets.instances().iter().enumerate() {
        let candidate_index = candidate.candidate_indices[set_index];
        let point = &instance.candidates().points[candidate_index];
        for (name, value) in &point.values {
            if symbols.insert(name.clone(), *value).is_some() {
                return Err(MacroSpecificationEvaluationError::DuplicateSymbol {
                    symbol: name.clone(),
                });
            }
        }
    }
    for outcome in &candidate.ac_outcomes {
        let testbench = macro_.exploration().testbenches()[outcome.testbench_index].name();
        insert_metrics(&mut symbols, testbench, outcome.outcome.metrics)?;
    }
    Ok(symbols)
}

fn insert_metrics(
    symbols: &mut HashMap<String, f64>,
    testbench: &str,
    metrics: AcMetrics,
) -> Result<(), MacroSpecificationEvaluationError> {
    for (name, value) in [
        ("dc_gain_db", metrics.dc_gain_db),
        ("bandwidth_3db_hz", metrics.bandwidth_3db_hz),
        ("unity_gain_hz", metrics.unity_gain_hz),
        ("phase_margin_deg", metrics.phase_margin_deg),
    ] {
        if let Some(value) = value {
            let symbol = format!("{testbench}.{name}");
            if symbols.insert(symbol.clone(), value).is_some() {
                return Err(MacroSpecificationEvaluationError::DuplicateSymbol { symbol });
            }
        }
    }
    Ok(())
}

fn selected_candidate_value(
    candidate_sets: &MacroCandidateSets,
    candidate: &MacroAcceptedCandidate,
    instance_path: &str,
    column: &str,
) -> Option<f64> {
    let position = candidate_sets
        .instances()
        .iter()
        .position(|instance| instance.instance_path() == instance_path)?;
    let index = *candidate.candidate_indices.get(position)?;
    candidate_sets
        .instances()
        .get(position)?
        .candidates()
        .points
        .get(index)?
        .get(column)
}

fn selected_metric(
    macro_: &Macro,
    candidate: &MacroAcceptedCandidate,
    testbench: &str,
    metric: AcMetric,
) -> Option<f64> {
    let testbench_index = macro_
        .exploration()
        .testbenches()
        .iter()
        .position(|definition| definition.name() == testbench)?;
    let metrics = candidate
        .ac_outcomes
        .iter()
        .find(|outcome| outcome.testbench_index == testbench_index)?
        .outcome
        .metrics;
    match metric {
        AcMetric::DcGainDb => metrics.dc_gain_db,
        AcMetric::Bandwidth3DbHz => metrics.bandwidth_3db_hz,
        AcMetric::UnityGainHz => metrics.unity_gain_hz,
        AcMetric::PhaseMarginDeg => metrics.phase_margin_deg,
    }
}

fn expression_symbols(expression: &str) -> Result<Vec<String>, MacroSpecificationEvaluationError> {
    let symbols = HashMap::new();
    let parser = ExpressionParser::new(expression, &symbols);
    parser.symbols().map_err(
        |reason| MacroSpecificationEvaluationError::InvalidExpression {
            expression: expression.to_owned(),
            reason,
        },
    )
}

pub(super) fn validate_numeric_expression(expression: &str) -> Result<(), String> {
    ExpressionParser::new(expression, &HashMap::new())
        .symbols()
        .map(|_| ())
}

struct ExpressionParser<'a> {
    expression: &'a str,
    bytes: &'a [u8],
    pos: usize,
    symbols: &'a HashMap<String, f64>,
    referenced: HashSet<String>,
}

impl<'a> ExpressionParser<'a> {
    fn new(expression: &'a str, symbols: &'a HashMap<String, f64>) -> Self {
        Self {
            expression,
            bytes: expression.as_bytes(),
            pos: 0,
            symbols,
            referenced: HashSet::new(),
        }
    }

    fn parse(mut self) -> Result<f64, String> {
        let value = self.parse_expression(true)?;
        self.finish()?;
        Ok(value)
    }

    fn symbols(mut self) -> Result<Vec<String>, String> {
        self.parse_expression(false)?;
        self.finish()?;
        let mut symbols = self.referenced.into_iter().collect::<Vec<_>>();
        symbols.sort();
        Ok(symbols)
    }

    fn finish(&mut self) -> Result<(), String> {
        self.skip_whitespace();
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(format!("unexpected token at byte {}", self.pos))
        }
    }

    fn parse_expression(&mut self, resolve: bool) -> Result<f64, String> {
        let mut value = self.parse_term(resolve)?;
        loop {
            self.skip_whitespace();
            if self.consume(b'+') {
                value += self.parse_term(resolve)?;
            } else if self.consume(b'-') {
                value -= self.parse_term(resolve)?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_term(&mut self, resolve: bool) -> Result<f64, String> {
        let mut value = self.parse_factor(resolve)?;
        loop {
            self.skip_whitespace();
            if self.consume(b'*') {
                value *= self.parse_factor(resolve)?;
            } else if self.consume(b'/') {
                value /= self.parse_factor(resolve)?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_factor(&mut self, resolve: bool) -> Result<f64, String> {
        self.skip_whitespace();
        if self.consume(b'-') {
            return Ok(-self.parse_factor(resolve)?);
        }
        if self.consume(b'+') {
            return self.parse_factor(resolve);
        }
        if self.consume(b'(') {
            let value = self.parse_expression(resolve)?;
            self.skip_whitespace();
            if !self.consume(b')') {
                return Err("missing closing ')'".to_owned());
            }
            return Ok(value);
        }
        if self.peek().is_some_and(is_number_start) {
            return self.parse_number();
        }
        if self.peek().is_some_and(is_identifier_start) {
            let symbol = self.parse_symbol_name();
            self.referenced.insert(symbol.to_owned());
            return if resolve {
                self.symbols
                    .get(symbol)
                    .copied()
                    .ok_or_else(|| format!("missing symbol '{symbol}'"))
            } else {
                Ok(0.0)
            };
        }
        Err(format!("expected value at byte {}", self.pos))
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
            let exponent = self.pos;
            while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                self.pos += 1;
            }
            if self.pos == exponent {
                return Err("missing exponent digits".to_owned());
            }
        }
        self.expression[start..self.pos]
            .parse()
            .map_err(|error| format!("invalid number: {error}"))
    }

    fn parse_symbol_name(&mut self) -> &'a str {
        let start = self.pos;
        while self.peek().is_some_and(is_identifier_body) {
            self.pos += 1;
        }
        &self.expression[start..self.pos]
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

/// Invalid specification declaration or runtime value resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroSpecificationEvaluationError {
    EmptyName {
        index: usize,
    },
    DuplicateName {
        name: String,
    },
    InvalidBounds {
        specification: String,
    },
    UnknownTestbench {
        specification: String,
        testbench: String,
    },
    MetricNotSelected {
        specification: String,
        testbench: String,
        metric: AcMetric,
    },
    InvalidExpression {
        expression: String,
        reason: String,
    },
    DependencyCycle {
        specification: String,
    },
    UnknownCandidateInstance {
        specification: String,
        instance_path: String,
    },
    MissingCandidateColumn {
        specification: String,
        instance_path: String,
        column: String,
    },
    UnavailableMetric {
        specification: String,
        testbench: String,
        metric: AcMetric,
    },
    DuplicateSymbol {
        symbol: String,
    },
    MissingSymbol {
        specification: String,
        symbol: String,
    },
    Expression {
        specification: String,
        expression: String,
        reason: String,
    },
    NonFiniteValue {
        specification: String,
    },
}

impl fmt::Display for MacroSpecificationEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName { index } => {
                write!(
                    formatter,
                    "specification at index {index} has an empty name"
                )
            }
            Self::DuplicateName { name } => {
                write!(
                    formatter,
                    "specification '{name}' is declared more than once"
                )
            }
            Self::InvalidBounds { specification } => write!(
                formatter,
                "specification '{specification}' has non-finite or inverted bounds"
            ),
            Self::UnknownTestbench {
                specification,
                testbench,
            } => write!(
                formatter,
                "specification '{specification}' references unknown testbench '{testbench}'"
            ),
            Self::MetricNotSelected {
                specification,
                testbench,
                metric,
            } => write!(
                formatter,
                "specification '{specification}' references {metric:?}, which testbench '{testbench}' does not select"
            ),
            Self::InvalidExpression { expression, reason } => {
                write!(
                    formatter,
                    "invalid specification expression '{expression}': {reason}"
                )
            }
            Self::DependencyCycle { specification } => write!(
                formatter,
                "specification dependency cycle reaches '{specification}'"
            ),
            Self::UnknownCandidateInstance {
                specification,
                instance_path,
            } => write!(
                formatter,
                "specification '{specification}' references unknown candidate instance '{instance_path}'"
            ),
            Self::MissingCandidateColumn {
                specification,
                instance_path,
                column,
            } => write!(
                formatter,
                "specification '{specification}' cannot find column '{column}' on instance '{instance_path}'"
            ),
            Self::UnavailableMetric {
                specification,
                testbench,
                metric,
            } => write!(
                formatter,
                "specification '{specification}' cannot read {metric:?} from testbench '{testbench}'"
            ),
            Self::DuplicateSymbol { symbol } => {
                write!(
                    formatter,
                    "specification context contains duplicate symbol '{symbol}'"
                )
            }
            Self::MissingSymbol {
                specification,
                symbol,
            } => write!(
                formatter,
                "specification '{specification}' references missing symbol '{symbol}'"
            ),
            Self::Expression {
                specification,
                expression,
                reason,
            } => write!(
                formatter,
                "specification '{specification}' could not evaluate '{expression}': {reason}"
            ),
            Self::NonFiniteValue { specification } => {
                write!(
                    formatter,
                    "specification '{specification}' produced a non-finite value"
                )
            }
        }
    }
}

impl Error for MacroSpecificationEvaluationError {}

#[cfg(test)]
mod tests {
    use super::ExpressionParser;
    use std::collections::HashMap;

    #[test]
    fn evaluates_basic_arithmetic_and_collects_symbols() {
        let expression = "(gain.stage1 + gain_stage2) / 2 - offset";
        let symbols = ExpressionParser::new(expression, &HashMap::new())
            .symbols()
            .unwrap();
        assert_eq!(symbols, ["gain.stage1", "gain_stage2", "offset"]);
        let values = HashMap::from([
            ("gain.stage1".to_owned(), 10.0),
            ("gain_stage2".to_owned(), 6.0),
            ("offset".to_owned(), 1.0),
        ]);
        assert_eq!(
            ExpressionParser::new(expression, &values).parse().unwrap(),
            7.0
        );
    }
}
