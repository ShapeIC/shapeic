//! Projection of accepted macro results into compact parent-visible candidates.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::analysis::AcMetric;
use crate::exploration::candidate::{CandidatePoint, CandidateSet, candidate_column_name};
use crate::netlist::names::compact_model_param_name;

use super::validation::validate_output_bindings;
use super::{
    Macro, MacroAcceptedCandidate, MacroExplorationResult, MacroOutputSource, MacroValidationError,
};

/// Parent-visible candidates projected from one immutable child result.
///
/// Only explicitly bound compact parameters and interface variables are copied
/// into the candidate rows. The complete child result is shared for recursive
/// sizing and operating-point provenance lookup.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroCandidateProjection {
    candidates: CandidateSet,
    interface_ports: Vec<String>,
    source_result: Arc<MacroExplorationResult>,
    accepted_indices: Vec<usize>,
}

impl MacroCandidateProjection {
    /// Returns the compact parent-visible candidate set.
    pub const fn candidates(&self) -> &CandidateSet {
        &self.candidates
    }

    /// Returns public ports whose projected columns carry interface values.
    pub fn interface_ports(&self) -> &[String] {
        &self.interface_ports
    }

    /// Returns the immutable child result retained for recursive provenance.
    pub fn source_result(&self) -> &Arc<MacroExplorationResult> {
        &self.source_result
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        CandidateSet,
        Vec<String>,
        Arc<MacroExplorationResult>,
        Vec<usize>,
    ) {
        (
            self.candidates,
            self.interface_ports,
            self.source_result,
            self.accepted_indices,
        )
    }
}

impl MacroExplorationResult {
    /// Projects this shared result for one named instance in a parent circuit.
    pub fn project(
        self: &Arc<Self>,
        macro_: &Macro,
        instance_path: impl Into<String>,
    ) -> Result<MacroCandidateProjection, MacroCandidateProjectionError> {
        project_result(Arc::clone(self), macro_, instance_path.into())
    }

    /// Consumes and projects this result for a single parent instance.
    pub fn into_projection(
        self,
        macro_: &Macro,
        instance_path: impl Into<String>,
    ) -> Result<MacroCandidateProjection, MacroCandidateProjectionError> {
        project_result(Arc::new(self), macro_, instance_path.into())
    }
}

fn project_result(
    result: Arc<MacroExplorationResult>,
    macro_: &Macro,
    instance_path: String,
) -> Result<MacroCandidateProjection, MacroCandidateProjectionError> {
    if result.macro_name() != macro_.name() {
        return Err(MacroCandidateProjectionError::MacroMismatch {
            result_macro: result.macro_name().to_owned(),
            definition_macro: macro_.name().to_owned(),
        });
    }
    if instance_path.trim().is_empty() {
        return Err(MacroCandidateProjectionError::EmptyInstancePath);
    }
    let mut binding_errors = Vec::new();
    validate_output_bindings(macro_, &mut binding_errors);
    if !binding_errors.is_empty() {
        return Err(MacroCandidateProjectionError::InvalidBindings {
            macro_name: macro_.name().to_owned(),
            errors: binding_errors,
        });
    }

    let exploration = macro_.exploration();
    let mut column_names = Vec::with_capacity(
        exploration.compact_outputs().len() + exploration.interface_bindings().len(),
    );
    let mut sources = Vec::with_capacity(column_names.capacity());
    let mut seen_columns = HashSet::with_capacity(column_names.capacity());
    for binding in exploration.compact_outputs() {
        let column = compact_model_param_name(binding.parameter(), &instance_path);
        if !seen_columns.insert(column.clone()) {
            return Err(MacroCandidateProjectionError::DuplicateProjectedColumn { column });
        }
        column_names.push(column);
        sources.push(binding.source());
    }

    let mut interface_ports = Vec::with_capacity(exploration.interface_bindings().len());
    for binding in exploration.interface_bindings() {
        let column = candidate_column_name(&instance_path, &binding.port().to_ascii_lowercase());
        if !seen_columns.insert(column.clone()) {
            return Err(MacroCandidateProjectionError::DuplicateProjectedColumn { column });
        }
        column_names.push(column);
        sources.push(binding.source());
        interface_ports.push(binding.port().to_owned());
    }

    let mut points = Vec::with_capacity(result.accepted().len());
    for (accepted_index, accepted) in result.accepted().iter().enumerate() {
        let values = column_names
            .iter()
            .zip(&sources)
            .map(|(column, source)| {
                let value = resolve_source(&result, accepted, source).map_err(|error| {
                    MacroCandidateProjectionError::ResolveSource {
                        accepted_index,
                        projected_column: column.clone(),
                        error,
                    }
                })?;
                Ok((column.clone(), value))
            })
            .collect::<Result<Vec<_>, MacroCandidateProjectionError>>()?;
        points.push(CandidatePoint::new(values));
    }

    Ok(MacroCandidateProjection {
        candidates: CandidateSet::new(instance_path, points),
        interface_ports,
        accepted_indices: (0..result.accepted().len()).collect(),
        source_result: result,
    })
}

fn resolve_source(
    result: &MacroExplorationResult,
    accepted: &MacroAcceptedCandidate,
    source: &MacroOutputSource,
) -> Result<f64, MacroOutputResolutionError> {
    let value = match source {
        MacroOutputSource::CandidateColumn {
            instance_path,
            column,
        } => result
            .selected_candidate(accepted, instance_path)
            .ok_or_else(|| MacroOutputResolutionError::MissingCandidateInstance {
                instance_path: instance_path.clone(),
            })?
            .get(column)
            .ok_or_else(|| MacroOutputResolutionError::MissingCandidateColumn {
                instance_path: instance_path.clone(),
                column: column.clone(),
            })?,
        MacroOutputSource::AcMetric { testbench, metric } => {
            let outcome = result.ac_outcome(accepted, testbench).ok_or_else(|| {
                MacroOutputResolutionError::MissingAcOutcome {
                    testbench: testbench.clone(),
                }
            })?;
            metric_value(outcome.metrics, *metric).ok_or_else(|| {
                MacroOutputResolutionError::UnavailableAcMetric {
                    testbench: testbench.clone(),
                    metric: *metric,
                }
            })?
        }
        MacroOutputSource::DcNodeVoltage { testbench } => result
            .dc_node_voltage_outcome(accepted, testbench)
            .ok_or_else(|| MacroOutputResolutionError::MissingDcNodeVoltageOutcome {
                testbench: testbench.clone(),
            })?
            .voltage_v(),
    };
    if !value.is_finite() {
        return Err(MacroOutputResolutionError::NonFiniteValue);
    }
    Ok(value)
}

fn metric_value(metrics: crate::analysis::AcMetrics, metric: AcMetric) -> Option<f64> {
    match metric {
        AcMetric::DcGainDb => metrics.dc_gain_db,
        AcMetric::Bandwidth3DbHz => metrics.bandwidth_3db_hz,
        AcMetric::UnityGainHz => metrics.unity_gain_hz,
        AcMetric::PhaseMarginDeg => metrics.phase_margin_deg,
    }
}

/// Errors produced while projecting accepted macro results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroCandidateProjectionError {
    MacroMismatch {
        result_macro: String,
        definition_macro: String,
    },
    EmptyInstancePath,
    InvalidBindings {
        macro_name: String,
        errors: Vec<MacroValidationError>,
    },
    DuplicateProjectedColumn {
        column: String,
    },
    ResolveSource {
        accepted_index: usize,
        projected_column: String,
        error: MacroOutputResolutionError,
    },
}

impl fmt::Display for MacroCandidateProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MacroMismatch {
                result_macro,
                definition_macro,
            } => write!(
                formatter,
                "cannot project result for macro '{result_macro}' using definition '{definition_macro}'"
            ),
            Self::EmptyInstancePath => {
                formatter.write_str("projected macro instance path is empty")
            }
            Self::InvalidBindings { macro_name, errors } => write!(
                formatter,
                "macro '{macro_name}' has {} invalid output binding definition(s)",
                errors.len()
            ),
            Self::DuplicateProjectedColumn { column } => {
                write!(
                    formatter,
                    "projected candidate column '{column}' is duplicated"
                )
            }
            Self::ResolveSource {
                accepted_index,
                projected_column,
                error,
            } => write!(
                formatter,
                "could not resolve projected column '{projected_column}' for accepted candidate {accepted_index}: {error}"
            ),
        }
    }
}

impl Error for MacroCandidateProjectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ResolveSource { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Errors produced while resolving one explicit output source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroOutputResolutionError {
    MissingCandidateInstance {
        instance_path: String,
    },
    MissingCandidateColumn {
        instance_path: String,
        column: String,
    },
    MissingAcOutcome {
        testbench: String,
    },
    MissingDcNodeVoltageOutcome {
        testbench: String,
    },
    UnavailableAcMetric {
        testbench: String,
        metric: AcMetric,
    },
    NonFiniteValue,
}

impl fmt::Display for MacroOutputResolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCandidateInstance { instance_path } => write!(
                formatter,
                "accepted result has no selected candidate for instance '{instance_path}'"
            ),
            Self::MissingCandidateColumn {
                instance_path,
                column,
            } => write!(
                formatter,
                "selected candidate for instance '{instance_path}' has no column '{column}'"
            ),
            Self::MissingAcOutcome { testbench } => {
                write!(
                    formatter,
                    "accepted result has no AC outcome for '{testbench}'"
                )
            }
            Self::MissingDcNodeVoltageOutcome { testbench } => write!(
                formatter,
                "accepted result has no DC node-voltage outcome for '{testbench}'"
            ),
            Self::UnavailableAcMetric { testbench, metric } => write!(
                formatter,
                "AC metric {} is unavailable for testbench '{testbench}'",
                metric.label()
            ),
            Self::NonFiniteValue => formatter.write_str("resolved output value is not finite"),
        }
    }
}

impl Error for MacroOutputResolutionError {}
