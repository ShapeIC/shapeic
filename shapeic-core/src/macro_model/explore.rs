use std::error::Error;
use std::fmt;

use crate::analysis::{AcCompletion, AcMetric, AdaptiveAcOutcome};
use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::exploration::candidate::CandidatePoint;

use super::{
    Macro, MacroAcCandidateAnalysisError, MacroCandidateBuildError, MacroCandidateCombinationError,
    MacroCandidateCombinationJoin, MacroCandidateSets, MacroCatalog, MacroExplorationInput,
    MacroRenderMode, MacroTestbenchPrepareError, PreparedMacroAcCandidateEvaluator,
    PreparedMacroAcCandidateEvaluatorError, build_macro_candidate_sets, prepare_macro_ac_testbench,
};

/// One accepted macro candidate and all AC outcomes evaluated for it.
#[derive(Debug, PartialEq)]
pub struct MacroAcceptedCandidate {
    candidate_indices: Vec<usize>,
    ac_outcomes: Vec<MacroAcTestbenchOutcome>,
}

impl MacroAcceptedCandidate {
    /// Returns one candidate index per explorable instance, in circuit order.
    pub fn candidate_indices(&self) -> &[usize] {
        &self.candidate_indices
    }

    /// Returns the successful testbench outcomes in macro declaration order.
    pub fn ac_outcomes(&self) -> &[MacroAcTestbenchOutcome] {
        &self.ac_outcomes
    }
}

/// Indexed AC outcome retained for an accepted macro candidate.
#[derive(Debug, PartialEq)]
pub struct MacroAcTestbenchOutcome {
    testbench_index: usize,
    outcome: AdaptiveAcOutcome,
}

impl MacroAcTestbenchOutcome {
    /// Returns the testbench position in macro declaration order.
    pub const fn testbench_index(&self) -> usize {
        self.testbench_index
    }

    /// Returns the complete adaptive AC result.
    pub const fn outcome(&self) -> &AdaptiveAcOutcome {
        &self.outcome
    }
}

/// AC rejection counts separated by the first failed specification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MacroAcRejectionCounts {
    dc_gain: usize,
    bandwidth_3db: usize,
    unity_gain: usize,
    phase_margin: usize,
}

impl MacroAcRejectionCounts {
    /// Returns the number rejected by the requested AC metric.
    pub const fn count(self, metric: AcMetric) -> usize {
        match metric {
            AcMetric::DcGainDb => self.dc_gain,
            AcMetric::Bandwidth3DbHz => self.bandwidth_3db,
            AcMetric::UnityGainHz => self.unity_gain,
            AcMetric::PhaseMarginDeg => self.phase_margin,
        }
    }

    /// Returns the total candidates rejected by this testbench.
    pub const fn total(self) -> usize {
        self.dc_gain + self.bandwidth_3db + self.unity_gain + self.phase_margin
    }

    fn record(&mut self, metric: AcMetric) {
        match metric {
            AcMetric::DcGainDb => self.dc_gain += 1,
            AcMetric::Bandwidth3DbHz => self.bandwidth_3db += 1,
            AcMetric::UnityGainHz => self.unity_gain += 1,
            AcMetric::PhaseMarginDeg => self.phase_margin += 1,
        }
    }
}

/// Evaluation and rejection statistics for one macro AC testbench.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroAcTestbenchStatistics {
    testbench: String,
    evaluated_candidates: usize,
    frequency_evaluations: usize,
    rejections: MacroAcRejectionCounts,
}

impl MacroAcTestbenchStatistics {
    fn new(testbench: impl Into<String>) -> Self {
        Self {
            testbench: testbench.into(),
            evaluated_candidates: 0,
            frequency_evaluations: 0,
            rejections: MacroAcRejectionCounts::default(),
        }
    }

    /// Returns the macro-local testbench name.
    pub fn testbench(&self) -> &str {
        &self.testbench
    }

    /// Returns how many compatible candidates reached this testbench.
    pub const fn evaluated_candidates(&self) -> usize {
        self.evaluated_candidates
    }

    /// Returns the total numerical frequency evaluations spent here.
    pub const fn frequency_evaluations(&self) -> usize {
        self.frequency_evaluations
    }

    /// Returns rejection counts classified by the first failed AC metric.
    pub const fn rejections(&self) -> MacroAcRejectionCounts {
        self.rejections
    }
}

/// Aggregate statistics for one macro candidate exploration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MacroExplorationStatistics {
    compatible_candidates: usize,
    accepted_candidates: usize,
    testbenches: Vec<MacroAcTestbenchStatistics>,
}

impl MacroExplorationStatistics {
    /// Returns the number of connectivity-compatible selections evaluated.
    pub const fn compatible_candidates(&self) -> usize {
        self.compatible_candidates
    }

    /// Returns the number of selections accepted by every testbench.
    pub const fn accepted_candidates(&self) -> usize {
        self.accepted_candidates
    }

    /// Returns the number rejected by the first failing testbench.
    pub const fn rejected_candidates(&self) -> usize {
        self.compatible_candidates - self.accepted_candidates
    }

    /// Returns per-testbench statistics in evaluation order.
    pub fn testbenches(&self) -> &[MacroAcTestbenchStatistics] {
        &self.testbenches
    }

    /// Finds statistics for one macro-local testbench.
    pub fn testbench(&self, name: &str) -> Option<&MacroAcTestbenchStatistics> {
        self.testbenches
            .iter()
            .find(|testbench| testbench.testbench == name)
    }

    /// Returns all numerical frequency evaluations across every testbench.
    pub fn frequency_evaluations(&self) -> usize {
        self.testbenches
            .iter()
            .map(MacroAcTestbenchStatistics::frequency_evaluations)
            .sum()
    }
}

/// Immutable accepted results, candidate provenance, and exploration statistics.
#[derive(Debug, PartialEq)]
pub struct MacroExplorationResult {
    macro_name: String,
    candidate_sets: MacroCandidateSets,
    accepted: Vec<MacroAcceptedCandidate>,
    statistics: MacroExplorationStatistics,
}

impl MacroExplorationResult {
    /// Returns the explored macro name.
    pub fn macro_name(&self) -> &str {
        &self.macro_name
    }

    /// Returns the filtered candidate sets retained for provenance lookup.
    pub const fn candidate_sets(&self) -> &MacroCandidateSets {
        &self.candidate_sets
    }

    /// Returns candidates accepted by every configured testbench.
    pub fn accepted(&self) -> &[MacroAcceptedCandidate] {
        &self.accepted
    }

    /// Returns aggregate and per-testbench exploration statistics.
    pub const fn statistics(&self) -> &MacroExplorationStatistics {
        &self.statistics
    }

    /// Resolves the selected candidate index for one instance path.
    pub fn selected_candidate_index(
        &self,
        accepted: &MacroAcceptedCandidate,
        instance_path: &str,
    ) -> Option<usize> {
        let instance_position = self
            .candidate_sets
            .instances()
            .iter()
            .position(|instance| instance.instance_path() == instance_path)?;
        accepted.candidate_indices.get(instance_position).copied()
    }

    /// Resolves the complete selected candidate point for sizing and voltage lookup.
    pub fn selected_candidate(
        &self,
        accepted: &MacroAcceptedCandidate,
        instance_path: &str,
    ) -> Option<&CandidatePoint> {
        let candidate_index = self.selected_candidate_index(accepted, instance_path)?;
        self.candidate_sets
            .instance(instance_path)?
            .candidates()
            .points
            .get(candidate_index)
    }

    /// Resolves the selected child result and accepted candidate for a compact
    /// submacro instance.
    pub fn selected_submacro<'a>(
        &'a self,
        accepted: &MacroAcceptedCandidate,
        instance_path: &str,
    ) -> Option<(&'a MacroExplorationResult, &'a MacroAcceptedCandidate)> {
        let candidate_index = self.selected_candidate_index(accepted, instance_path)?;
        let provenance = self
            .candidate_sets
            .instance(instance_path)?
            .compact_provenance
            .as_ref()?;
        let child_accepted_index = *provenance.accepted_indices.get(candidate_index)?;
        let child_result = provenance.source_result.as_ref();
        let child_accepted = child_result.accepted.get(child_accepted_index)?;
        Some((child_result, child_accepted))
    }

    /// Finds one accepted AC outcome by its macro-local testbench name.
    pub fn ac_outcome<'a>(
        &self,
        accepted: &'a MacroAcceptedCandidate,
        testbench: &str,
    ) -> Option<&'a AdaptiveAcOutcome> {
        let testbench_index = self
            .statistics
            .testbenches
            .iter()
            .position(|statistics| statistics.testbench == testbench)?;
        accepted
            .ac_outcomes
            .iter()
            .find(|outcome| outcome.testbench_index == testbench_index)
            .map(MacroAcTestbenchOutcome::outcome)
    }
}

/// Errors produced while preparing or executing macro-wide AC exploration.
#[derive(Debug)]
pub enum MacroAcExplorationError {
    CandidateCombination(MacroCandidateCombinationError),
    PrepareTestbench {
        testbench: String,
        error: MacroTestbenchPrepareError,
    },
    PrepareCandidateEvaluator {
        testbench: String,
        error: Box<PreparedMacroAcCandidateEvaluatorError>,
    },
    AnalyzeCandidate {
        testbench: String,
        candidate_indices: Vec<usize>,
        error: Box<MacroAcCandidateAnalysisError>,
    },
    InconsistentOutcome {
        testbench: String,
        candidate_indices: Vec<usize>,
        reason: &'static str,
    },
}

impl fmt::Display for MacroAcExplorationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateCombination(error) => error.fmt(formatter),
            Self::PrepareTestbench { testbench, error } => write!(
                formatter,
                "could not prepare macro AC testbench '{testbench}': {error}"
            ),
            Self::PrepareCandidateEvaluator { testbench, error } => write!(
                formatter,
                "could not prepare candidate bindings for AC testbench '{testbench}': {error}"
            ),
            Self::AnalyzeCandidate {
                testbench,
                candidate_indices,
                error,
            } => write!(
                formatter,
                "could not analyze candidate selection {candidate_indices:?} with AC testbench '{testbench}': {error}"
            ),
            Self::InconsistentOutcome {
                testbench,
                candidate_indices,
                reason,
            } => write!(
                formatter,
                "AC testbench '{testbench}' returned an inconsistent outcome for candidate selection {candidate_indices:?}: {reason}"
            ),
        }
    }
}

impl Error for MacroAcExplorationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CandidateCombination(error) => Some(error),
            Self::PrepareTestbench { error, .. } => Some(error),
            Self::PrepareCandidateEvaluator { error, .. } => Some(error.as_ref()),
            Self::AnalyzeCandidate { error, .. } => Some(error.as_ref()),
            Self::InconsistentOutcome { .. } => None,
        }
    }
}

/// Errors produced by the complete macro exploration entry point.
#[derive(Debug)]
pub enum MacroExplorationError {
    /// Primitive or compact-submacro candidate construction failed.
    BuildCandidates(MacroCandidateBuildError),
    /// Candidate combination, testbench preparation, or AC evaluation failed.
    Ac(MacroAcExplorationError),
}

impl fmt::Display for MacroExplorationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BuildCandidates(error) => error.fmt(formatter),
            Self::Ac(error) => error.fmt(formatter),
        }
    }
}

impl Error for MacroExplorationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::BuildCandidates(error) => Some(error),
            Self::Ac(error) => Some(error),
        }
    }
}

impl From<MacroCandidateBuildError> for MacroExplorationError {
    fn from(error: MacroCandidateBuildError) -> Self {
        Self::BuildCandidates(error)
    }
}

impl From<MacroAcExplorationError> for MacroExplorationError {
    fn from(error: MacroAcExplorationError) -> Self {
        Self::Ac(error)
    }
}

impl Macro {
    /// Builds local candidate sets, combines them by connectivity, and runs all
    /// AC testbenches declared by this macro.
    pub fn explore(
        &self,
        primitive_catalog: &PrimitiveCatalog,
        macro_catalog: &MacroCatalog,
        input: MacroExplorationInput<'_>,
    ) -> Result<MacroExplorationResult, MacroExplorationError> {
        let candidate_sets = build_macro_candidate_sets(self, primitive_catalog, input)?;
        explore_macro_ac_candidates(self, primitive_catalog, macro_catalog, candidate_sets)
            .map_err(Into::into)
    }
}

/// Executes all macro-owned AC testbenches over connectivity-compatible candidates.
///
/// Testbenches are prepared once in declaration order. A candidate rejected by
/// one testbench is not evaluated by later testbenches, and only candidates
/// accepted by every testbench retain their outcomes.
pub fn explore_macro_ac_candidates(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
    candidate_sets: MacroCandidateSets,
) -> Result<MacroExplorationResult, MacroAcExplorationError> {
    let testbench_names = macro_
        .exploration()
        .testbenches()
        .iter()
        .map(|testbench| testbench.name().to_owned())
        .collect::<Vec<_>>();
    let (accepted, statistics) = {
        let mut combinations =
            MacroCandidateCombinationJoin::new(macro_, primitive_catalog, &candidate_sets)
                .map_err(MacroAcExplorationError::CandidateCombination)?;
        let has_empty_candidate_set = candidate_sets
            .candidate_sets()
            .any(|candidates| candidates.points.is_empty());

        if has_empty_candidate_set {
            (
                Vec::new(),
                MacroExplorationStatistics::new(&testbench_names),
            )
        } else {
            let candidate_refs = candidate_sets.candidate_sets().collect::<Vec<_>>();
            let mut evaluators = Vec::with_capacity(macro_.exploration().testbenches().len());
            for testbench in macro_.exploration().testbenches() {
                let prepared = prepare_macro_ac_testbench(
                    macro_,
                    testbench,
                    primitive_catalog,
                    macro_catalog,
                    MacroRenderMode::CompactSubmacros,
                )
                .map_err(|error| MacroAcExplorationError::PrepareTestbench {
                    testbench: testbench.name().to_owned(),
                    error,
                })?;
                let evaluator = PreparedMacroAcCandidateEvaluator::new(prepared, &candidate_refs)
                    .map_err(|error| {
                    MacroAcExplorationError::PrepareCandidateEvaluator {
                        testbench: testbench.name().to_owned(),
                        error: Box::new(error),
                    }
                })?;
                evaluators.push(evaluator);
            }
            evaluate_candidate_combinations(
                &mut combinations,
                &testbench_names,
                |testbench_index, candidate_indices| {
                    evaluators[testbench_index].analyze(candidate_indices)
                },
            )
            .map_err(|error| match error {
                CandidateEvaluationLoopError::Analysis {
                    testbench_index,
                    candidate_indices,
                    error,
                } => MacroAcExplorationError::AnalyzeCandidate {
                    testbench: testbench_names[testbench_index].clone(),
                    candidate_indices,
                    error: Box::new(error),
                },
                CandidateEvaluationLoopError::InconsistentOutcome {
                    testbench_index,
                    candidate_indices,
                    reason,
                } => MacroAcExplorationError::InconsistentOutcome {
                    testbench: testbench_names[testbench_index].clone(),
                    candidate_indices,
                    reason,
                },
            })?
        }
    };

    Ok(MacroExplorationResult {
        macro_name: macro_.name().to_owned(),
        candidate_sets,
        accepted,
        statistics,
    })
}

impl MacroExplorationStatistics {
    fn new(testbench_names: &[String]) -> Self {
        Self {
            testbenches: testbench_names
                .iter()
                .cloned()
                .map(MacroAcTestbenchStatistics::new)
                .collect(),
            ..Self::default()
        }
    }
}

#[derive(Debug)]
enum CandidateEvaluationLoopError<E> {
    Analysis {
        testbench_index: usize,
        candidate_indices: Vec<usize>,
        error: E,
    },
    InconsistentOutcome {
        testbench_index: usize,
        candidate_indices: Vec<usize>,
        reason: &'static str,
    },
}

fn evaluate_candidate_combinations<E>(
    combinations: &mut MacroCandidateCombinationJoin<'_>,
    testbench_names: &[String],
    mut analyze: impl FnMut(usize, &[usize]) -> Result<AdaptiveAcOutcome, E>,
) -> Result<
    (Vec<MacroAcceptedCandidate>, MacroExplorationStatistics),
    CandidateEvaluationLoopError<E>,
> {
    let mut statistics = MacroExplorationStatistics::new(testbench_names);
    let mut accepted = Vec::new();

    while let Some(candidate_indices) = combinations.next_selection() {
        statistics.compatible_candidates += 1;
        let mut ac_outcomes = Vec::with_capacity(testbench_names.len());
        let mut rejected = false;

        for testbench_index in 0..testbench_names.len() {
            let outcome = analyze(testbench_index, candidate_indices).map_err(|error| {
                CandidateEvaluationLoopError::Analysis {
                    testbench_index,
                    candidate_indices: candidate_indices.to_vec(),
                    error,
                }
            })?;
            let testbench_statistics = &mut statistics.testbenches[testbench_index];
            testbench_statistics.evaluated_candidates += 1;
            testbench_statistics.frequency_evaluations += outcome.frequency_evaluations;

            if let Some(metric) = rejection_metric(&outcome).map_err(|reason| {
                CandidateEvaluationLoopError::InconsistentOutcome {
                    testbench_index,
                    candidate_indices: candidate_indices.to_vec(),
                    reason,
                }
            })? {
                testbench_statistics.rejections.record(metric);
                rejected = true;
                break;
            }
            ac_outcomes.push(MacroAcTestbenchOutcome {
                testbench_index,
                outcome,
            });
        }

        if !rejected {
            accepted.push(MacroAcceptedCandidate {
                candidate_indices: candidate_indices.to_vec(),
                ac_outcomes,
            });
        }
    }

    statistics.accepted_candidates = accepted.len();
    debug_assert_eq!(
        statistics.compatible_candidates,
        statistics.accepted_candidates
            + statistics
                .testbenches
                .iter()
                .map(|testbench| testbench.rejections.total())
                .sum::<usize>()
    );
    Ok((accepted, statistics))
}

fn rejection_metric(outcome: &AdaptiveAcOutcome) -> Result<Option<AcMetric>, &'static str> {
    match (outcome.completion, outcome.targets.passed()) {
        (AcCompletion::Complete, true) => Ok(None),
        (AcCompletion::PrunedAfter(_), true) => {
            Err("analysis was pruned although all evaluated targets passed")
        }
        (AcCompletion::PrunedAfter(metric), false) => {
            if outcome
                .targets
                .failures()
                .iter()
                .any(|failure| failure.metric == metric)
            {
                Ok(Some(metric))
            } else {
                Err("pruning metric is not present in the target failures")
            }
        }
        (AcCompletion::Complete, false) => outcome
            .targets
            .failures()
            .first()
            .map(|failure| Some(failure.metric))
            .ok_or("failed target assessment contains no failures"),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::analysis::{
        AcMetricSet, AcMetrics, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets,
        TargetAssessment,
    };
    use crate::circuit::Circuit;
    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::exploration::filter::CandidateFilter;
    use crate::exploration::filter::CandidateFilterReport;
    use crate::macro_model::{
        CompactMacroInstanceExplorationInput, MacroAcTestbench, MacroCompactOutputBinding,
        MacroExplorationInput, MacroInstanceCandidateSet, MacroInterfaceBinding, MacroOutputSource,
        MacroPort, MacroPortRole, MacroRenderMode, build_macro_candidate_sets,
        render_small_signal_netlist,
    };
    use crate::netlist::names::{compact_model_param_name, small_signal_param_name};
    use crate::primitive::build::{PrimitiveBuildSpec, SweepMode};
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};
    use crate::primitive::small_signal::{SmallSignalBranch, SmallSignalModel};
    use crate::testbench::{AcAnalysis, TransferFunction};
    use shapeic_lut::{MosCapacitanceMatrix, MosExtrinsicCapacitances};

    use super::*;

    fn analysis() -> AcAnalysis {
        AcAnalysis::new(
            TransferFunction::new("VIN", "VOUT"),
            AdaptiveAcConfig {
                min_frequency_hz: 1.0,
                max_frequency_hz: 1.0e6,
                coarse_points_per_decade: 4,
                crossing_relative_tolerance: 1.0e-4,
                max_refinement_steps: 16,
                retain_samples: false,
            },
            AdaptiveAcPolicy {
                mode: AnalysisMode::Prune,
                targets: AnalysisTargets {
                    min_dc_gain_db: Some(10.0),
                    min_bandwidth_3db_hz: None,
                    min_unity_gain_hz: None,
                    min_phase_margin_deg: None,
                },
                metrics: AcMetricSet::from_metric(AcMetric::DcGainDb),
            },
        )
    }

    fn primitive_catalog() -> PrimitiveCatalog {
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(PrimitiveManifest {
            name: "gain".to_owned(),
            version: "1.0".to_owned(),
            description: None,
            subckt_name: "gain".to_owned(),
            pins: vec![
                Pin {
                    name: "VIN".to_owned(),
                    role: PinRole::Input,
                },
                Pin {
                    name: "VOUT".to_owned(),
                    role: PinRole::Output,
                },
                Pin {
                    name: "VSS".to_owned(),
                    role: PinRole::Supply,
                },
            ],
            files: PrimitiveFiles {
                netlist: "gain.spice".to_owned(),
                build: None,
                symbol: None,
            },
            small_signal: Some(SmallSignalModel::new(vec![SmallSignalBranch::new(
                "m1", "VOUT", "VIN", "VSS", "VSS",
            )])),
            transistor_type: Some("nmos".to_owned()),
            layout_params: None,
            lut_config: None,
            build: Some(PrimitiveBuildSpec {
                inputs: Vec::new(),
                sweep_mode: SweepMode::Aligned,
                derived: Vec::new(),
                lut: Vec::new(),
                columns: Vec::new(),
            }),
        });
        catalog
    }

    fn macro_() -> Macro {
        Macro::new(
            "gain_stage",
            vec![
                MacroPort::new("VIN", MacroPortRole::Input),
                MacroPort::new("VOUT", MacroPortRole::Output),
                MacroPort::new("VSS", MacroPortRole::Ground),
            ],
            Circuit::builder()
                .primitive(
                    "xcore",
                    "gain",
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .resistor("rload", "VOUT", "VSS", 1.0e4)
                .build(),
            Circuit::builder()
                .vccs("gm", "VOUT", "VSS", "VIN", "VSS", "gm_eq")
                .resistor("gain", "VOUT", "VSS", "gain_eq")
                .build(),
        )
        .with_ac_testbench(MacroAcTestbench::from_spice(
            "gain",
            "Vinput VIN VSS 1\n.end\n",
            analysis(),
        ))
        .with_compact_output(MacroCompactOutputBinding::new(
            "gm_eq",
            MacroOutputSource::candidate_column(
                "xcore",
                small_signal_param_name("gm", "xcore", "m1"),
            ),
        ))
        .with_compact_output(MacroCompactOutputBinding::new(
            "gain_eq",
            MacroOutputSource::ac_metric("gain", AcMetric::DcGainDb),
        ))
        .with_interface_binding(MacroInterfaceBinding::new(
            "VOUT",
            MacroOutputSource::candidate_column("xcore", "xcore.vout"),
        ))
    }

    fn candidate(gm: f64) -> CandidatePoint {
        let mut values = vec![
            (small_signal_param_name("gm", "xcore", "m1"), gm),
            (small_signal_param_name("ro", "xcore", "m1"), 1.0e5),
            ("xcore.vout".to_owned(), 1.0),
            ("xcore.width_m1".to_owned(), gm * 1.0e6),
            (small_signal_param_name("width", "xcore", "m1"), 8.0e-6),
            (small_signal_param_name("length", "xcore", "m1"), 1.0e-6),
            (small_signal_param_name("nf", "xcore", "m1"), 2.0),
        ];
        values.extend(
            MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
                .into_iter()
                .chain(MosExtrinsicCapacitances::PARAMETERS)
                .map(|parameter| (small_signal_param_name(parameter, "xcore", "m1"), 1.0e-15)),
        );
        CandidatePoint::new(values)
    }

    fn candidates() -> MacroCandidateSets {
        MacroCandidateSets {
            instances: vec![MacroInstanceCandidateSet {
                instance_path: "xcore".to_owned(),
                kind: super::super::MacroExplorationInstanceKind::Primitive,
                candidates: CandidateSet::new("xcore", vec![candidate(1.0e-3), candidate(1.0e-5)]),
                filter_report: CandidateFilterReport::default(),
                compact_provenance: None,
            }],
        }
    }

    #[test]
    fn evaluates_testbenches_in_order_and_stops_after_the_first_rejection() {
        let macro_ = macro_();
        let primitives = primitive_catalog();
        let candidates = candidates();
        let mut combinations =
            MacroCandidateCombinationJoin::new(&macro_, &primitives, &candidates).unwrap();
        let testbench_names = vec!["first".to_owned(), "second".to_owned()];
        let mut calls = Vec::new();

        let (accepted, statistics) = evaluate_candidate_combinations(
            &mut combinations,
            &testbench_names,
            |testbench_index, candidate_indices| {
                calls.push((testbench_index, candidate_indices.to_vec()));
                if testbench_index == 0 && candidate_indices[0] == 1 {
                    Ok::<_, ()>(rejected_outcome(AcMetric::DcGainDb, 2))
                } else {
                    Ok::<_, ()>(passing_outcome(3))
                }
            },
        )
        .unwrap();

        assert_eq!(calls, [(0, vec![0]), (1, vec![0]), (0, vec![1]),]);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].candidate_indices(), [0]);
        assert_eq!(accepted[0].ac_outcomes().len(), 2);
        assert_eq!(statistics.compatible_candidates(), 2);
        assert_eq!(statistics.accepted_candidates(), 1);
        assert_eq!(statistics.rejected_candidates(), 1);
        assert_eq!(statistics.testbenches()[0].evaluated_candidates(), 2);
        assert_eq!(
            statistics.testbenches()[0]
                .rejections()
                .count(AcMetric::DcGainDb),
            1
        );
        assert_eq!(statistics.testbenches()[1].evaluated_candidates(), 1);
        assert_eq!(statistics.frequency_evaluations(), 8);
    }

    fn passing_outcome(frequency_evaluations: usize) -> AdaptiveAcOutcome {
        AdaptiveAcOutcome {
            metrics: AcMetrics::default(),
            requested_metrics: AcMetricSet::ALL,
            evaluated_metrics: AcMetricSet::ALL,
            targets: TargetAssessment::default(),
            completion: AcCompletion::Complete,
            samples: Vec::new(),
            frequency_evaluations,
        }
    }

    fn rejected_outcome(metric: AcMetric, frequency_evaluations: usize) -> AdaptiveAcOutcome {
        let targets = match metric {
            AcMetric::DcGainDb => AnalysisTargets {
                min_dc_gain_db: Some(1.0),
                min_bandwidth_3db_hz: None,
                min_unity_gain_hz: None,
                min_phase_margin_deg: None,
            },
            AcMetric::Bandwidth3DbHz => AnalysisTargets {
                min_dc_gain_db: None,
                min_bandwidth_3db_hz: Some(1.0),
                min_unity_gain_hz: None,
                min_phase_margin_deg: None,
            },
            AcMetric::UnityGainHz => AnalysisTargets {
                min_dc_gain_db: None,
                min_bandwidth_3db_hz: None,
                min_unity_gain_hz: Some(1.0),
                min_phase_margin_deg: None,
            },
            AcMetric::PhaseMarginDeg => AnalysisTargets {
                min_dc_gain_db: None,
                min_bandwidth_3db_hz: None,
                min_unity_gain_hz: None,
                min_phase_margin_deg: Some(1.0),
            },
        }
        .assess_metric(metric, Some(0.0));
        AdaptiveAcOutcome {
            metrics: AcMetrics::default(),
            requested_metrics: AcMetricSet::ALL,
            evaluated_metrics: AcMetricSet::ALL,
            targets,
            completion: AcCompletion::PrunedAfter(metric),
            samples: Vec::new(),
            frequency_evaluations,
        }
    }

    #[test]
    #[ignore = "requires the external Symbolica runtime used by numerical MNA preparation"]
    fn explores_compatible_candidates_and_keeps_only_accepted_provenance() {
        let macro_ = macro_();
        let primitives = primitive_catalog();
        let macros = MacroCatalog::from_macros([macro_.clone()]).unwrap();

        let result =
            explore_macro_ac_candidates(&macro_, &primitives, &macros, candidates()).unwrap();

        assert_eq!(result.statistics().compatible_candidates(), 2);
        assert_eq!(result.statistics().accepted_candidates(), 1);
        assert_eq!(result.statistics().rejected_candidates(), 1);
        let testbench = result.statistics().testbench("gain").unwrap();
        assert_eq!(testbench.evaluated_candidates(), 2);
        assert_eq!(testbench.rejections().count(AcMetric::DcGainDb), 1);
        let accepted = &result.accepted()[0];
        assert_eq!(result.selected_candidate_index(accepted, "xcore"), Some(0));
        assert_eq!(
            result
                .selected_candidate(accepted, "xcore")
                .and_then(|candidate| candidate.get("gm__xcore__m1")),
            Some(1.0e-3)
        );
        assert!(
            result
                .ac_outcome(accepted, "gain")
                .and_then(|outcome| outcome.metrics.dc_gain_db)
                .is_some_and(|gain| gain >= 10.0)
        );
    }

    #[test]
    fn classifies_complete_full_insight_failures_by_first_metric() {
        let metrics = AcMetrics {
            dc_gain_db: Some(0.0),
            bandwidth_3db_hz: None,
            unity_gain_hz: None,
            phase_margin_deg: None,
        };
        let targets = AnalysisTargets {
            min_dc_gain_db: Some(10.0),
            min_bandwidth_3db_hz: Some(1.0),
            min_unity_gain_hz: None,
            min_phase_margin_deg: None,
        }
        .assess_all(&metrics);
        let outcome = AdaptiveAcOutcome {
            metrics,
            requested_metrics: AcMetricSet::ALL,
            evaluated_metrics: AcMetricSet::ALL,
            targets,
            completion: AcCompletion::Complete,
            samples: Vec::new(),
            frequency_evaluations: 0,
        };

        assert_eq!(rejection_metric(&outcome), Ok(Some(AcMetric::DcGainDb)));
    }

    #[test]
    fn rejects_inconsistent_pruned_outcomes() {
        let outcome = AdaptiveAcOutcome {
            metrics: AcMetrics::default(),
            requested_metrics: AcMetricSet::ALL,
            evaluated_metrics: AcMetricSet::ALL,
            targets: TargetAssessment::default(),
            completion: AcCompletion::PrunedAfter(AcMetric::DcGainDb),
            samples: Vec::new(),
            frequency_evaluations: 0,
        };

        assert!(rejection_metric(&outcome).is_err());
    }

    #[test]
    fn projects_explicit_outputs_and_preserves_filtered_child_provenance() {
        let child_macro = macro_();
        let mut first_outcome = passing_outcome(0);
        first_outcome.metrics.dc_gain_db = Some(40.0);
        let mut second_outcome = passing_outcome(0);
        second_outcome.metrics.dc_gain_db = Some(20.0);
        let child_result = Arc::new(MacroExplorationResult {
            macro_name: child_macro.name().to_owned(),
            candidate_sets: candidates(),
            accepted: vec![
                MacroAcceptedCandidate {
                    candidate_indices: vec![0],
                    ac_outcomes: vec![MacroAcTestbenchOutcome {
                        testbench_index: 0,
                        outcome: first_outcome,
                    }],
                },
                MacroAcceptedCandidate {
                    candidate_indices: vec![1],
                    ac_outcomes: vec![MacroAcTestbenchOutcome {
                        testbench_index: 0,
                        outcome: second_outcome,
                    }],
                },
            ],
            statistics: MacroExplorationStatistics {
                compatible_candidates: 2,
                accepted_candidates: 2,
                testbenches: vec![MacroAcTestbenchStatistics::new("gain")],
            },
        });

        let projection = child_result.project(&child_macro, "xchild").unwrap();
        assert_eq!(projection.candidates().points.len(), 2);
        assert_eq!(
            projection.candidates().points[0].get(&compact_model_param_name("gm_eq", "xchild")),
            Some(1.0e-3)
        );
        assert_eq!(
            projection.candidates().points[1].get(&compact_model_param_name("gain_eq", "xchild")),
            Some(20.0)
        );
        assert_eq!(
            projection.candidates().points[0].get("xchild.vout"),
            Some(1.0)
        );
        assert!(
            projection.candidates().points[0]
                .get("xcore.width_m1")
                .is_none()
        );
        assert_eq!(projection.interface_ports(), ["VOUT"]);

        let parent = Macro::new(
            "parent",
            vec![
                MacroPort::new("VIN", MacroPortRole::Input),
                MacroPort::new("VOUT", MacroPortRole::Output),
                MacroPort::new("VSS", MacroPortRole::Ground),
            ],
            Circuit::builder()
                .macro_instance(
                    "xchild",
                    child_macro.name(),
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .macro_instance(
                    "xother",
                    child_macro.name(),
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .build(),
            Circuit::builder()
                .resistor("constant", "VOUT", "VSS", 1.0)
                .build(),
        );
        let mut input = MacroExplorationInput::new();
        input
            .register_compact_macro_instance(
                "xchild",
                CompactMacroInstanceExplorationInput::from_projection(
                    projection,
                    vec![
                        CandidateFilter::at_most(
                            compact_model_param_name("gm_eq", "xchild"),
                            1.0e-4,
                        )
                        .unwrap(),
                    ],
                ),
            )
            .unwrap();
        input
            .register_compact_macro_instance(
                "xother",
                CompactMacroInstanceExplorationInput::from_projection(
                    child_result.project(&child_macro, "xother").unwrap(),
                    Vec::new(),
                ),
            )
            .unwrap();
        let parent_sets =
            build_macro_candidate_sets(&parent, &PrimitiveCatalog::new(), input).unwrap();
        let combination_plan =
            MacroCandidateCombinationJoin::new(&parent, &PrimitiveCatalog::new(), &parent_sets)
                .unwrap();
        assert_eq!(combination_plan.plan().equalities().len(), 1);
        let parent_result = MacroExplorationResult {
            macro_name: parent.name().to_owned(),
            candidate_sets: parent_sets,
            accepted: vec![MacroAcceptedCandidate {
                candidate_indices: vec![0, 0],
                ac_outcomes: Vec::new(),
            }],
            statistics: MacroExplorationStatistics {
                compatible_candidates: 1,
                accepted_candidates: 1,
                testbenches: Vec::new(),
            },
        };

        let (resolved_child, resolved_accepted) = parent_result
            .selected_submacro(&parent_result.accepted()[0], "xchild")
            .unwrap();
        assert!(Arc::ptr_eq(
            parent_result.candidate_sets.instances[0]
                .compact_provenance
                .as_ref()
                .map(|provenance| &provenance.source_result)
                .unwrap(),
            &child_result
        ));
        assert_eq!(
            resolved_child.selected_candidate_index(resolved_accepted, "xcore"),
            Some(1)
        );
    }

    #[test]
    #[ignore = "requires the external Symbolica runtime used by numerical MNA preparation"]
    fn explores_a_child_then_uses_only_its_compact_model_in_the_parent() {
        let primitives = primitive_catalog();
        let child = Macro::new(
            "hierarchical_child",
            vec![
                MacroPort::new("VIN", MacroPortRole::Input),
                MacroPort::new("VOUT", MacroPortRole::Output),
                MacroPort::new("VSS", MacroPortRole::Ground),
            ],
            Circuit::builder()
                .primitive(
                    "xcore",
                    "gain",
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .resistor("rload", "VOUT", "VSS", 1.0e4)
                .build(),
            Circuit::builder()
                .vccs("gm", "VOUT", "VSS", "VIN", "VSS", "gm_eq")
                .resistor("ro", "VOUT", "VSS", "ro_eq")
                .build(),
        )
        .with_ac_testbench(MacroAcTestbench::from_spice(
            "gain",
            "Vinput VIN VSS 1\n.end\n",
            analysis(),
        ))
        .with_compact_output(MacroCompactOutputBinding::new(
            "gm_eq",
            MacroOutputSource::candidate_column(
                "xcore",
                small_signal_param_name("gm", "xcore", "m1"),
            ),
        ))
        .with_compact_output(MacroCompactOutputBinding::new(
            "ro_eq",
            MacroOutputSource::candidate_column(
                "xcore",
                small_signal_param_name("ro", "xcore", "m1"),
            ),
        ))
        .with_interface_binding(MacroInterfaceBinding::new(
            "VOUT",
            MacroOutputSource::candidate_column("xcore", "xcore.vout"),
        ));
        let parent = Macro::new(
            "hierarchical_parent",
            vec![
                MacroPort::new("VIN", MacroPortRole::Input),
                MacroPort::new("VOUT", MacroPortRole::Output),
                MacroPort::new("VSS", MacroPortRole::Ground),
            ],
            Circuit::builder()
                .macro_instance(
                    "xchild",
                    child.name(),
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .build(),
            Circuit::builder()
                .resistor("rout", "VOUT", "VSS", 1.0)
                .build(),
        )
        .with_ac_testbench(MacroAcTestbench::from_spice(
            "gain",
            "Vinput VIN VSS 1\n.end\n",
            analysis(),
        ));
        let macros = MacroCatalog::from_macros([child.clone(), parent.clone()]).unwrap();

        let child_result =
            explore_macro_ac_candidates(&child, &primitives, &macros, candidates()).unwrap();
        assert_eq!(child_result.statistics().accepted_candidates(), 1);
        let projection = child_result.into_projection(&child, "xchild").unwrap();

        let rendered_parent = render_small_signal_netlist(
            &parent,
            &primitives,
            &macros,
            MacroRenderMode::CompactSubmacros,
        )
        .unwrap();
        assert!(
            rendered_parent
                .source()
                .contains("G_xchild__gm VOUT VSS VIN VSS gm_eq__xchild")
        );
        assert!(
            rendered_parent
                .source()
                .contains("R_xchild__ro VOUT VSS ro_eq__xchild")
        );
        assert!(rendered_parent.primitive_branches().is_empty());
        assert!(!rendered_parent.source().contains("xcore"));

        let mut parent_input = MacroExplorationInput::new();
        parent_input
            .register_compact_macro_instance(
                "xchild",
                CompactMacroInstanceExplorationInput::from_projection(projection, Vec::new()),
            )
            .unwrap();
        let parent_result = parent.explore(&primitives, &macros, parent_input).unwrap();
        assert_eq!(parent_result.statistics().accepted_candidates(), 1);

        let parent_accepted = &parent_result.accepted()[0];
        let (resolved_child, child_accepted) = parent_result
            .selected_submacro(parent_accepted, "xchild")
            .unwrap();
        let original = resolved_child
            .selected_candidate(child_accepted, "xcore")
            .unwrap();
        assert_eq!(
            original.get(&small_signal_param_name("width", "xcore", "m1")),
            Some(8.0e-6)
        );
        assert_eq!(
            original.get(&small_signal_param_name("length", "xcore", "m1")),
            Some(1.0e-6)
        );
        assert_eq!(
            original.get(&small_signal_param_name("nf", "xcore", "m1")),
            Some(2.0)
        );
    }
}
