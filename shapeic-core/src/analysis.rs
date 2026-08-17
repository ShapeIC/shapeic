use std::error::Error;
use std::fmt;

use num_complex::Complex64;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AcMetrics {
    pub dc_gain_db: Option<f64>,
    pub bandwidth_3db_hz: Option<f64>,
    pub unity_gain_hz: Option<f64>,
    pub phase_margin_deg: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalysisTargets {
    pub min_dc_gain_db: Option<f64>,
    pub min_bandwidth_3db_hz: Option<f64>,
    pub min_unity_gain_hz: Option<f64>,
    pub min_phase_margin_deg: Option<f64>,
}

impl AnalysisTargets {
    pub const NONE: Self = Self {
        min_dc_gain_db: None,
        min_bandwidth_3db_hz: None,
        min_unity_gain_hz: None,
        min_phase_margin_deg: None,
    };

    pub fn validate(self) -> Result<(), InvalidTarget> {
        validate_target(AcMetric::DcGainDb, self.min_dc_gain_db, false)?;
        validate_target(AcMetric::Bandwidth3DbHz, self.min_bandwidth_3db_hz, true)?;
        validate_target(AcMetric::UnityGainHz, self.min_unity_gain_hz, true)?;
        validate_target(AcMetric::PhaseMarginDeg, self.min_phase_margin_deg, false)
    }

    pub fn assess_dc_gain(self, dc_gain_db: Option<f64>) -> TargetAssessment {
        self.assess_metric(AcMetric::DcGainDb, dc_gain_db)
    }

    pub fn assess_metric(self, metric: AcMetric, actual: Option<f64>) -> TargetAssessment {
        let minimum = match metric {
            AcMetric::DcGainDb => self.min_dc_gain_db,
            AcMetric::Bandwidth3DbHz => self.min_bandwidth_3db_hz,
            AcMetric::UnityGainHz => self.min_unity_gain_hz,
            AcMetric::PhaseMarginDeg => self.min_phase_margin_deg,
        };
        let mut assessment = TargetAssessment::default();
        assessment.check(metric, minimum, actual);
        assessment
    }

    pub fn assess_frequency_metrics(self, metrics: &AcMetrics) -> TargetAssessment {
        let mut assessment = TargetAssessment::default();
        assessment.check(
            AcMetric::Bandwidth3DbHz,
            self.min_bandwidth_3db_hz,
            metrics.bandwidth_3db_hz,
        );
        assessment.check(
            AcMetric::UnityGainHz,
            self.min_unity_gain_hz,
            metrics.unity_gain_hz,
        );
        assessment.check(
            AcMetric::PhaseMarginDeg,
            self.min_phase_margin_deg,
            metrics.phase_margin_deg,
        );
        assessment
    }

    pub fn assess_all(self, metrics: &AcMetrics) -> TargetAssessment {
        let mut assessment = self.assess_dc_gain(metrics.dc_gain_db);
        assessment
            .failures
            .extend(self.assess_frequency_metrics(metrics).failures);
        assessment
    }
}

impl Default for AnalysisTargets {
    fn default() -> Self {
        Self::NONE
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalysisMode {
    Prune,
    FullInsight,
}

impl AnalysisMode {
    pub fn should_continue(self, assessment: &TargetAssessment) -> bool {
        self == Self::FullInsight || assessment.passed()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcMetric {
    DcGainDb,
    Bandwidth3DbHz,
    UnityGainHz,
    PhaseMarginDeg,
}

impl AcMetric {
    pub const fn label(self) -> &'static str {
        match self {
            Self::DcGainDb => "GainDC",
            Self::Bandwidth3DbHz => "f3dB",
            Self::UnityGainHz => "UGF",
            Self::PhaseMarginDeg => "PM",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AcMetricSet(u8);

impl AcMetricSet {
    pub const EMPTY: Self = Self(0);
    pub const ALL: Self = Self(0b1111);

    pub const fn from_metric(metric: AcMetric) -> Self {
        Self(metric.bit())
    }

    pub const fn with(self, metric: AcMetric) -> Self {
        Self(self.0 | metric.bit())
    }

    pub const fn contains(self, metric: AcMetric) -> bool {
        self.0 & metric.bit() != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl AcMetric {
    const ALL: [Self; 4] = [
        Self::DcGainDb,
        Self::Bandwidth3DbHz,
        Self::UnityGainHz,
        Self::PhaseMarginDeg,
    ];

    const fn bit(self) -> u8 {
        match self {
            Self::DcGainDb => 1 << 0,
            Self::Bandwidth3DbHz => 1 << 1,
            Self::UnityGainHz => 1 << 2,
            Self::PhaseMarginDeg => 1 << 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetFailure {
    pub metric: AcMetric,
    pub minimum: f64,
    pub actual: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TargetAssessment {
    failures: Vec<TargetFailure>,
}

impl TargetAssessment {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn failures(&self) -> &[TargetFailure] {
        &self.failures
    }

    pub fn extend(&mut self, other: Self) {
        self.failures.extend(other.failures);
    }

    fn check(&mut self, metric: AcMetric, minimum: Option<f64>, actual: Option<f64>) {
        let Some(minimum) = minimum else {
            return;
        };
        let actual = actual.filter(|value| value.is_finite());
        if actual.is_none_or(|value| value < minimum) {
            self.failures.push(TargetFailure {
                metric,
                minimum,
                actual,
            });
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveAcConfig {
    pub min_frequency_hz: f64,
    pub max_frequency_hz: f64,
    pub coarse_points_per_decade: usize,
    pub crossing_relative_tolerance: f64,
    pub max_refinement_steps: usize,
    pub retain_samples: bool,
}

impl AdaptiveAcConfig {
    pub fn validate(self) -> Result<(), InvalidAdaptiveAcConfig> {
        let valid = self.min_frequency_hz.is_finite()
            && self.min_frequency_hz > 0.0
            && self.max_frequency_hz.is_finite()
            && self.max_frequency_hz > self.min_frequency_hz
            && self.coarse_points_per_decade > 0
            && self.crossing_relative_tolerance.is_finite()
            && self.crossing_relative_tolerance > 0.0
            && self.max_refinement_steps > 0;
        if valid {
            Ok(())
        } else {
            Err(InvalidAdaptiveAcConfig)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveAcPolicy {
    pub mode: AnalysisMode,
    pub targets: AnalysisTargets,
    pub metrics: AcMetricSet,
}

impl AdaptiveAcPolicy {
    pub const COMPLETE: Self = Self {
        mode: AnalysisMode::FullInsight,
        targets: AnalysisTargets::NONE,
        metrics: AcMetricSet::ALL,
    };

    pub fn validate(self) -> Result<(), InvalidAdaptiveAcPolicy> {
        if self.metrics.is_empty() {
            return Err(InvalidAdaptiveAcPolicy::EmptyMetricSelection);
        }
        self.targets
            .validate()
            .map_err(InvalidAdaptiveAcPolicy::InvalidTarget)?;

        for metric in AcMetric::ALL {
            if target_for_metric(self.targets, metric).is_some() && !self.metrics.contains(metric) {
                return Err(InvalidAdaptiveAcPolicy::TargetNotSelected { metric });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcCompletion {
    Complete,
    PrunedAfter(AcMetric),
}

impl AcCompletion {
    pub const fn stopped_after(self) -> Option<AcMetric> {
        match self {
            Self::Complete => None,
            Self::PrunedAfter(metric) => Some(metric),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveAcSample {
    pub frequency_hz: f64,
    pub response: Complex64,
    pub gain_db: f64,
    pub phase_deg: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveAcOutcome {
    pub metrics: AcMetrics,
    pub requested_metrics: AcMetricSet,
    pub evaluated_metrics: AcMetricSet,
    pub targets: TargetAssessment,
    pub completion: AcCompletion,
    pub samples: Vec<AdaptiveAcSample>,
    pub frequency_evaluations: usize,
}

impl AdaptiveAcOutcome {
    pub const fn metric_requested(&self, metric: AcMetric) -> bool {
        self.requested_metrics.contains(metric)
    }

    pub const fn metric_evaluated(&self, metric: AcMetric) -> bool {
        self.evaluated_metrics.contains(metric)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidAdaptiveAcConfig;

impl fmt::Display for InvalidAdaptiveAcConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid adaptive AC configuration")
    }
}

impl Error for InvalidAdaptiveAcConfig {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InvalidAdaptiveAcPolicy {
    EmptyMetricSelection,
    InvalidTarget(InvalidTarget),
    TargetNotSelected { metric: AcMetric },
}

impl fmt::Display for InvalidAdaptiveAcPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMetricSelection => {
                formatter.write_str("adaptive AC policy must select at least one metric")
            }
            Self::InvalidTarget(error) => error.fmt(formatter),
            Self::TargetNotSelected { metric } => write!(
                formatter,
                "adaptive AC target {} requires selecting the same metric",
                metric.label()
            ),
        }
    }
}

impl Error for InvalidAdaptiveAcPolicy {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidTarget(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum AdaptiveAcError<E> {
    InvalidConfig(InvalidAdaptiveAcConfig),
    InvalidPolicy(InvalidAdaptiveAcPolicy),
    Evaluation { frequency_hz: f64, source: E },
    NonFiniteResponse { frequency_hz: f64 },
}

impl<E: fmt::Display> fmt::Display for AdaptiveAcError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => error.fmt(formatter),
            Self::InvalidPolicy(error) => error.fmt(formatter),
            Self::Evaluation {
                frequency_hz,
                source,
            } => write!(
                formatter,
                "AC evaluation failed at {frequency_hz:.6e} Hz: {source}"
            ),
            Self::NonFiniteResponse { frequency_hz } => {
                write!(
                    formatter,
                    "AC response is not finite at {frequency_hz:.6e} Hz"
                )
            }
        }
    }
}

impl<E: Error + 'static> Error for AdaptiveAcError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConfig(error) => Some(error),
            Self::InvalidPolicy(error) => Some(error),
            Self::Evaluation { source, .. } => Some(source),
            Self::NonFiniteResponse { .. } => None,
        }
    }
}

pub fn analyze_adaptive_ac<E, F>(
    config: AdaptiveAcConfig,
    policy: AdaptiveAcPolicy,
    mut evaluate_loop_response: F,
) -> Result<AdaptiveAcOutcome, AdaptiveAcError<E>>
where
    F: FnMut(f64) -> Result<Complex64, E>,
{
    config.validate().map_err(AdaptiveAcError::InvalidConfig)?;
    policy.validate().map_err(AdaptiveAcError::InvalidPolicy)?;

    let needs_dc = policy.metrics.contains(AcMetric::DcGainDb)
        || policy.metrics.contains(AcMetric::Bandwidth3DbHz);
    let exact_dc = if needs_dc {
        Some(
            evaluate_loop_response(0.0).map_err(|source| AdaptiveAcError::Evaluation {
                frequency_hz: 0.0,
                source,
            })?,
        )
    } else {
        None
    };
    let mut sampler = AdaptiveSampler::new(config, &mut evaluate_loop_response);
    let dc_gain_db = match exact_dc {
        Some(response) if finite_response(response) => Some(magnitude_db(response)),
        Some(_) => Some(sampler.sample(config.min_frequency_hz)?.gain_db),
        None => None,
    };
    let mut metrics = AcMetrics::default();
    let mut targets = TargetAssessment::default();
    let mut evaluated_metrics = AcMetricSet::EMPTY;

    if policy.metrics.contains(AcMetric::DcGainDb) {
        metrics.dc_gain_db = dc_gain_db;
        evaluated_metrics = evaluated_metrics.with(AcMetric::DcGainDb);
        let assessment = policy.targets.assess_dc_gain(dc_gain_db);
        let prune = should_prune(policy.mode, &assessment);
        targets.extend(assessment);
        if prune {
            return Ok(sampler.finish(
                metrics,
                policy.metrics,
                evaluated_metrics,
                targets,
                AcCompletion::PrunedAfter(AcMetric::DcGainDb),
            ));
        }
    }

    if policy.metrics.contains(AcMetric::Bandwidth3DbHz) {
        let bandwidth_3db_hz = sampler
            .first_downward_crossing(
                dc_gain_db.expect("bandwidth selection must evaluate the DC reference") - 3.0,
            )?
            .map(|crossing| crossing.frequency_hz);
        metrics.bandwidth_3db_hz = bandwidth_3db_hz;
        evaluated_metrics = evaluated_metrics.with(AcMetric::Bandwidth3DbHz);
        let assessment = policy
            .targets
            .assess_metric(AcMetric::Bandwidth3DbHz, bandwidth_3db_hz);
        let prune = should_prune(policy.mode, &assessment);
        targets.extend(assessment);
        if prune {
            return Ok(sampler.finish(
                metrics,
                policy.metrics,
                evaluated_metrics,
                targets,
                AcCompletion::PrunedAfter(AcMetric::Bandwidth3DbHz),
            ));
        }
    }

    let needs_unity = policy.metrics.contains(AcMetric::UnityGainHz)
        || policy.metrics.contains(AcMetric::PhaseMarginDeg);
    let unity = if needs_unity {
        sampler.first_downward_crossing(0.0)?
    } else {
        None
    };

    if policy.metrics.contains(AcMetric::UnityGainHz) {
        let unity_gain_hz = unity.map(|crossing| crossing.frequency_hz);
        metrics.unity_gain_hz = unity_gain_hz;
        evaluated_metrics = evaluated_metrics.with(AcMetric::UnityGainHz);
        let assessment = policy
            .targets
            .assess_metric(AcMetric::UnityGainHz, unity_gain_hz);
        let prune = should_prune(policy.mode, &assessment);
        targets.extend(assessment);
        if prune {
            return Ok(sampler.finish(
                metrics,
                policy.metrics,
                evaluated_metrics,
                targets,
                AcCompletion::PrunedAfter(AcMetric::UnityGainHz),
            ));
        }
    }

    let mut completion = AcCompletion::Complete;
    if policy.metrics.contains(AcMetric::PhaseMarginDeg) {
        let phase_margin_deg =
            unity.map(|crossing| 180.0 + sampler.phase_at(crossing.frequency_hz));
        metrics.phase_margin_deg = phase_margin_deg;
        evaluated_metrics = evaluated_metrics.with(AcMetric::PhaseMarginDeg);
        let assessment = policy
            .targets
            .assess_metric(AcMetric::PhaseMarginDeg, phase_margin_deg);
        let prune = should_prune(policy.mode, &assessment);
        targets.extend(assessment);
        if prune {
            completion = AcCompletion::PrunedAfter(AcMetric::PhaseMarginDeg);
        }
    }

    Ok(sampler.finish(
        metrics,
        policy.metrics,
        evaluated_metrics,
        targets,
        completion,
    ))
}

fn should_prune(mode: AnalysisMode, assessment: &TargetAssessment) -> bool {
    mode == AnalysisMode::Prune && !assessment.passed()
}

#[derive(Clone, Copy)]
struct Crossing {
    frequency_hz: f64,
}

struct AdaptiveSampler<'a, E, F>
where
    F: FnMut(f64) -> Result<Complex64, E>,
{
    config: AdaptiveAcConfig,
    evaluate: &'a mut F,
    samples: Vec<AdaptiveAcSample>,
}

impl<'a, E, F> AdaptiveSampler<'a, E, F>
where
    F: FnMut(f64) -> Result<Complex64, E>,
{
    fn new(config: AdaptiveAcConfig, evaluate: &'a mut F) -> Self {
        Self {
            config,
            evaluate,
            samples: Vec::new(),
        }
    }

    fn sample(&mut self, frequency_hz: f64) -> Result<AdaptiveAcSample, AdaptiveAcError<E>> {
        if let Some(sample) = self
            .samples
            .iter()
            .find(|sample| sample.frequency_hz.to_bits() == frequency_hz.to_bits())
        {
            return Ok(*sample);
        }
        let response =
            (self.evaluate)(frequency_hz).map_err(|source| AdaptiveAcError::Evaluation {
                frequency_hz,
                source,
            })?;
        if !finite_response(response) {
            return Err(AdaptiveAcError::NonFiniteResponse { frequency_hz });
        }
        let sample = AdaptiveAcSample {
            frequency_hz,
            response,
            gain_db: magnitude_db(response),
            phase_deg: 0.0,
        };
        self.samples.push(sample);
        Ok(sample)
    }

    fn first_downward_crossing(
        &mut self,
        target_db: f64,
    ) -> Result<Option<Crossing>, AdaptiveAcError<E>> {
        let frequencies = coarse_frequencies(self.config);
        let mut lower = self.sample(frequencies[0])?;
        for frequency in frequencies.into_iter().skip(1) {
            let upper = self.sample(frequency)?;
            if lower.gain_db >= target_db
                && upper.gain_db <= target_db
                && lower.gain_db != upper.gain_db
            {
                return self.refine_crossing(lower, upper, target_db).map(Some);
            }
            lower = upper;
        }
        Ok(None)
    }

    fn refine_crossing(
        &mut self,
        mut lower: AdaptiveAcSample,
        mut upper: AdaptiveAcSample,
        target_db: f64,
    ) -> Result<Crossing, AdaptiveAcError<E>> {
        for _ in 0..self.config.max_refinement_steps {
            if upper.frequency_hz / lower.frequency_hz - 1.0
                <= self.config.crossing_relative_tolerance
            {
                break;
            }
            let middle_frequency = (lower.frequency_hz * upper.frequency_hz).sqrt();
            let middle = self.sample(middle_frequency)?;
            if middle.gain_db >= target_db {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        let weight = (target_db - lower.gain_db) / (upper.gain_db - lower.gain_db);
        let log_frequency = lower.frequency_hz.log10()
            + weight * (upper.frequency_hz.log10() - lower.frequency_hz.log10());
        let frequency_hz = 10.0_f64.powf(log_frequency);
        self.sample(frequency_hz)?;
        Ok(Crossing { frequency_hz })
    }

    fn phase_at(&mut self, frequency_hz: f64) -> f64 {
        self.samples
            .sort_by(|left, right| left.frequency_hz.total_cmp(&right.frequency_hz));
        let mut previous = None;
        let mut requested_phase = None;
        for sample in self
            .samples
            .iter_mut()
            .filter(|sample| sample.frequency_hz <= frequency_hz)
        {
            let mut phase = sample.response.arg().to_degrees();
            if previous.is_none() && phase > 90.0 {
                phase -= 360.0;
            }
            if let Some(previous) = previous {
                while phase - previous > 180.0 {
                    phase -= 360.0;
                }
                while phase - previous < -180.0 {
                    phase += 360.0;
                }
            }
            sample.phase_deg = phase;
            previous = Some(phase);
            if sample.frequency_hz.to_bits() == frequency_hz.to_bits() {
                requested_phase = Some(phase);
            }
        }
        requested_phase.expect("crossing frequency must have a cached sample")
    }

    fn finish(
        mut self,
        metrics: AcMetrics,
        requested_metrics: AcMetricSet,
        evaluated_metrics: AcMetricSet,
        targets: TargetAssessment,
        completion: AcCompletion,
    ) -> AdaptiveAcOutcome {
        self.samples
            .sort_by(|left, right| left.frequency_hz.total_cmp(&right.frequency_hz));
        fill_unwrapped_phases(&mut self.samples);
        let frequency_evaluations = self.samples.len();
        if !self.config.retain_samples {
            self.samples.clear();
        }
        AdaptiveAcOutcome {
            metrics,
            requested_metrics,
            evaluated_metrics,
            targets,
            completion,
            samples: self.samples,
            frequency_evaluations,
        }
    }
}

fn coarse_frequencies(config: AdaptiveAcConfig) -> Vec<f64> {
    let decades = (config.max_frequency_hz / config.min_frequency_hz).log10();
    let intervals = (decades * config.coarse_points_per_decade as f64).ceil() as usize;
    (0..=intervals)
        .map(|index| {
            if index == intervals {
                config.max_frequency_hz
            } else {
                let exponent = index as f64 / config.coarse_points_per_decade as f64;
                (config.min_frequency_hz * 10.0_f64.powf(exponent)).min(config.max_frequency_hz)
            }
        })
        .collect()
}

fn fill_unwrapped_phases(samples: &mut [AdaptiveAcSample]) {
    let mut previous = None;
    for sample in samples {
        let mut phase = sample.response.arg().to_degrees();
        if previous.is_none() && phase > 90.0 {
            phase -= 360.0;
        }
        if let Some(previous) = previous {
            while phase - previous > 180.0 {
                phase -= 360.0;
            }
            while phase - previous < -180.0 {
                phase += 360.0;
            }
        }
        sample.phase_deg = phase;
        previous = Some(phase);
    }
}

fn finite_response(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn magnitude_db(value: Complex64) -> f64 {
    10.0 * value.norm_sqr().log10()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvalidTarget {
    pub metric: AcMetric,
    pub value: f64,
}

impl fmt::Display for InvalidTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid minimum target {}={}: expected {}",
            self.metric.label(),
            self.value,
            if matches!(
                self.metric,
                AcMetric::Bandwidth3DbHz | AcMetric::UnityGainHz
            ) {
                "a finite positive frequency"
            } else {
                "a finite value"
            }
        )
    }
}

impl Error for InvalidTarget {}

fn validate_target(
    metric: AcMetric,
    minimum: Option<f64>,
    positive: bool,
) -> Result<(), InvalidTarget> {
    let Some(value) = minimum else {
        return Ok(());
    };
    if !value.is_finite() || (positive && value <= 0.0) {
        Err(InvalidTarget { metric, value })
    } else {
        Ok(())
    }
}

fn target_for_metric(targets: AnalysisTargets, metric: AcMetric) -> Option<f64> {
    match metric {
        AcMetric::DcGainDb => targets.min_dc_gain_db,
        AcMetric::Bandwidth3DbHz => targets.min_bandwidth_3db_hz,
        AcMetric::UnityGainHz => targets.min_unity_gain_hz,
        AcMetric::PhaseMarginDeg => targets.min_phase_margin_deg,
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use num_complex::Complex64;

    use super::{
        AcCompletion, AcMetric, AcMetricSet, AcMetrics, AdaptiveAcConfig, AdaptiveAcError,
        AdaptiveAcPolicy, AnalysisMode, AnalysisTargets, InvalidAdaptiveAcPolicy, TargetAssessment,
        analyze_adaptive_ac,
    };

    const TARGETS: AnalysisTargets = AnalysisTargets {
        min_dc_gain_db: Some(20.0),
        min_bandwidth_3db_hz: Some(100.0e6),
        min_unity_gain_hz: Some(1.0e9),
        min_phase_margin_deg: Some(45.0),
    };

    fn passing_metrics() -> AcMetrics {
        AcMetrics {
            dc_gain_db: Some(20.0),
            bandwidth_3db_hz: Some(100.0e6),
            unity_gain_hz: Some(1.0e9),
            phase_margin_deg: Some(45.0),
        }
    }

    fn adaptive_config() -> AdaptiveAcConfig {
        AdaptiveAcConfig {
            min_frequency_hz: 1.0,
            max_frequency_hz: 1.0e9,
            coarse_points_per_decade: 4,
            crossing_relative_tolerance: 0.005,
            max_refinement_steps: 32,
            retain_samples: true,
        }
    }

    fn one_pole_response(frequency_hz: f64, dc_gain: f64, pole_hz: f64) -> Complex64 {
        dc_gain / Complex64::new(1.0, frequency_hz / pole_hz)
    }

    fn selected_policy(metrics: AcMetricSet) -> AdaptiveAcPolicy {
        AdaptiveAcPolicy {
            mode: AnalysisMode::FullInsight,
            targets: AnalysisTargets::NONE,
            metrics,
        }
    }

    #[test]
    fn disabled_targets_always_pass() {
        let missing = AcMetrics {
            dc_gain_db: None,
            bandwidth_3db_hz: None,
            unity_gain_hz: None,
            phase_margin_deg: None,
        };

        assert!(AnalysisTargets::NONE.assess_all(&missing).passed());
    }

    #[test]
    fn values_equal_to_the_minimum_pass() {
        assert!(TARGETS.assess_all(&passing_metrics()).passed());
    }

    #[test]
    fn reports_below_minimum_and_unavailable_metrics() {
        let metrics = AcMetrics {
            dc_gain_db: Some(19.0),
            bandwidth_3db_hz: Some(90.0e6),
            unity_gain_hz: None,
            phase_margin_deg: Some(f64::NAN),
        };
        let assessment = TARGETS.assess_all(&metrics);

        assert_eq!(assessment.failures().len(), 4);
        assert_eq!(assessment.failures()[0].metric, AcMetric::DcGainDb);
        assert_eq!(assessment.failures()[0].actual, Some(19.0));
        assert_eq!(assessment.failures()[1].metric, AcMetric::Bandwidth3DbHz);
        assert_eq!(assessment.failures()[2].metric, AcMetric::UnityGainHz);
        assert_eq!(assessment.failures()[2].actual, None);
        assert_eq!(assessment.failures()[3].metric, AcMetric::PhaseMarginDeg);
        assert_eq!(assessment.failures()[3].actual, None);
    }

    #[test]
    fn validates_finite_targets_and_positive_frequencies() {
        assert!(TARGETS.validate().is_ok());

        let invalid_frequency = AnalysisTargets {
            min_unity_gain_hz: Some(0.0),
            ..AnalysisTargets::NONE
        };
        assert_eq!(
            invalid_frequency.validate().unwrap_err().metric,
            AcMetric::UnityGainHz
        );

        let invalid_gain = AnalysisTargets {
            min_dc_gain_db: Some(f64::INFINITY),
            ..AnalysisTargets::NONE
        };
        assert_eq!(
            invalid_gain.validate().unwrap_err().metric,
            AcMetric::DcGainDb
        );
    }

    #[test]
    fn prune_stops_on_failure_while_full_insight_continues() {
        let assessment = TargetAssessment {
            failures: vec![super::TargetFailure {
                metric: AcMetric::DcGainDb,
                minimum: 20.0,
                actual: Some(19.0),
            }],
        };

        assert!(!AnalysisMode::Prune.should_continue(&assessment));
        assert!(AnalysisMode::FullInsight.should_continue(&assessment));
    }

    #[test]
    fn adaptive_analysis_finds_one_pole_metrics_within_tolerance() {
        let pole_hz = 1.0e6;
        let dc_gain = 10.0;
        let outcome =
            analyze_adaptive_ac(adaptive_config(), AdaptiveAcPolicy::COMPLETE, |frequency| {
                Ok::<_, Infallible>(one_pole_response(frequency, dc_gain, pole_hz))
            })
            .unwrap();

        let expected_unity = pole_hz * (dc_gain * dc_gain - 1.0_f64).sqrt();
        let expected_phase_margin = 180.0 - (expected_unity / pole_hz).atan().to_degrees();
        assert!((outcome.metrics.dc_gain_db.unwrap() - 20.0).abs() < 1.0e-12);
        assert!((outcome.metrics.bandwidth_3db_hz.unwrap() / pole_hz - 1.0).abs() <= 0.005);
        assert!((outcome.metrics.unity_gain_hz.unwrap() / expected_unity - 1.0).abs() <= 0.005);
        assert!((outcome.metrics.phase_margin_deg.unwrap() - expected_phase_margin).abs() <= 0.5);
        assert_eq!(outcome.completion, AcCompletion::Complete);
        assert!(outcome.frequency_evaluations < 80);
    }

    #[test]
    fn dc_only_evaluates_zero_frequency_without_a_sweep() {
        let mut calls = Vec::new();
        let outcome = analyze_adaptive_ac(
            adaptive_config(),
            selected_policy(AcMetricSet::from_metric(AcMetric::DcGainDb)),
            |frequency| {
                calls.push(frequency);
                Ok::<_, Infallible>(Complex64::new(10.0, 0.0))
            },
        )
        .unwrap();

        assert_eq!(calls, vec![0.0]);
        assert_eq!(outcome.metrics.dc_gain_db, Some(20.0));
        assert_eq!(outcome.metrics.bandwidth_3db_hz, None);
        assert!(outcome.metric_requested(AcMetric::DcGainDb));
        assert!(outcome.metric_evaluated(AcMetric::DcGainDb));
        assert!(!outcome.metric_requested(AcMetric::Bandwidth3DbHz));
        assert_eq!(outcome.frequency_evaluations, 0);
    }

    #[test]
    fn dc_only_uses_the_minimum_frequency_as_a_non_finite_fallback() {
        let mut calls = Vec::new();
        let outcome = analyze_adaptive_ac(
            adaptive_config(),
            selected_policy(AcMetricSet::from_metric(AcMetric::DcGainDb)),
            |frequency| {
                calls.push(frequency);
                Ok::<_, Infallible>(if frequency == 0.0 {
                    Complex64::new(f64::NAN, f64::NAN)
                } else {
                    Complex64::new(10.0, 0.0)
                })
            },
        )
        .unwrap();

        assert_eq!(calls, vec![0.0, adaptive_config().min_frequency_hz]);
        assert_eq!(outcome.metrics.dc_gain_db, Some(20.0));
        assert_eq!(outcome.frequency_evaluations, 1);
    }

    #[test]
    fn bandwidth_only_uses_dc_internally_without_publishing_it() {
        let mut calls = Vec::new();
        let outcome = analyze_adaptive_ac(
            adaptive_config(),
            selected_policy(AcMetricSet::from_metric(AcMetric::Bandwidth3DbHz)),
            |frequency| {
                calls.push(frequency);
                Ok::<_, Infallible>(one_pole_response(frequency, 10.0, 1.0e6))
            },
        )
        .unwrap();

        assert_eq!(calls[0], 0.0);
        assert_eq!(outcome.metrics.dc_gain_db, None);
        assert!(outcome.metrics.bandwidth_3db_hz.is_some());
        assert!(!outcome.metric_evaluated(AcMetric::DcGainDb));
        assert!(outcome.metric_evaluated(AcMetric::Bandwidth3DbHz));
    }

    #[test]
    fn unity_gain_only_skips_dc_and_bandwidth() {
        let mut calls = Vec::new();
        let outcome = analyze_adaptive_ac(
            adaptive_config(),
            selected_policy(AcMetricSet::from_metric(AcMetric::UnityGainHz)),
            |frequency| {
                calls.push(frequency);
                Ok::<_, Infallible>(one_pole_response(frequency, 10.0, 1.0e6))
            },
        )
        .unwrap();

        assert!(calls.iter().all(|frequency| *frequency > 0.0));
        assert_eq!(outcome.metrics.dc_gain_db, None);
        assert_eq!(outcome.metrics.bandwidth_3db_hz, None);
        assert!(outcome.metrics.unity_gain_hz.is_some());
        assert_eq!(outcome.metrics.phase_margin_deg, None);
        assert!(outcome.metric_evaluated(AcMetric::UnityGainHz));
    }

    #[test]
    fn phase_margin_only_keeps_unity_gain_internal() {
        let outcome = analyze_adaptive_ac(
            adaptive_config(),
            selected_policy(AcMetricSet::from_metric(AcMetric::PhaseMarginDeg)),
            |frequency| Ok::<_, Infallible>(one_pole_response(frequency, 10.0, 1.0e6)),
        )
        .unwrap();

        assert_eq!(outcome.metrics.dc_gain_db, None);
        assert_eq!(outcome.metrics.bandwidth_3db_hz, None);
        assert_eq!(outcome.metrics.unity_gain_hz, None);
        assert!(outcome.metrics.phase_margin_deg.is_some());
        assert!(!outcome.metric_requested(AcMetric::UnityGainHz));
        assert!(!outcome.metric_evaluated(AcMetric::UnityGainHz));
        assert!(outcome.metric_evaluated(AcMetric::PhaseMarginDeg));
    }

    #[test]
    fn rejects_empty_selection_and_targets_for_unselected_metrics() {
        assert_eq!(
            selected_policy(AcMetricSet::EMPTY).validate(),
            Err(InvalidAdaptiveAcPolicy::EmptyMetricSelection)
        );

        let policy = AdaptiveAcPolicy {
            mode: AnalysisMode::Prune,
            targets: AnalysisTargets {
                min_unity_gain_hz: Some(1.0e6),
                ..AnalysisTargets::NONE
            },
            metrics: AcMetricSet::from_metric(AcMetric::DcGainDb),
        };
        assert_eq!(
            policy.validate(),
            Err(InvalidAdaptiveAcPolicy::TargetNotSelected {
                metric: AcMetric::UnityGainHz
            })
        );

        let mut evaluated = false;
        let error = analyze_adaptive_ac(
            adaptive_config(),
            selected_policy(AcMetricSet::EMPTY),
            |_| {
                evaluated = true;
                Ok::<_, Infallible>(Complex64::new(1.0, 0.0))
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            AdaptiveAcError::InvalidPolicy(InvalidAdaptiveAcPolicy::EmptyMetricSelection)
        ));
        assert!(!evaluated);
    }

    #[test]
    fn adaptive_prune_stops_after_the_first_failed_frequency_target() {
        let targets = AnalysisTargets {
            min_bandwidth_3db_hz: Some(2.0e6),
            min_unity_gain_hz: Some(100.0e6),
            ..AnalysisTargets::NONE
        };
        let policy = AdaptiveAcPolicy {
            mode: AnalysisMode::Prune,
            targets,
            metrics: AcMetricSet::ALL,
        };
        let outcome = analyze_adaptive_ac(adaptive_config(), policy, |frequency| {
            Ok::<_, Infallible>(one_pole_response(frequency, 10.0, 1.0e6))
        })
        .unwrap();

        assert_eq!(
            outcome.completion,
            AcCompletion::PrunedAfter(AcMetric::Bandwidth3DbHz)
        );
        assert!(outcome.metrics.bandwidth_3db_hz.is_some());
        assert!(outcome.metrics.unity_gain_hz.is_none());
        assert!(outcome.metric_evaluated(AcMetric::Bandwidth3DbHz));
        assert!(!outcome.metric_evaluated(AcMetric::UnityGainHz));
        assert_eq!(outcome.targets.failures().len(), 1);
        assert_eq!(
            outcome.targets.failures()[0].metric,
            AcMetric::Bandwidth3DbHz
        );
    }

    #[test]
    fn adaptive_complete_mode_reports_missing_crossings() {
        let mut calls = 0;
        let outcome = analyze_adaptive_ac(adaptive_config(), AdaptiveAcPolicy::COMPLETE, |_| {
            calls += 1;
            Ok::<_, Infallible>(Complex64::new(10.0, 0.0))
        })
        .unwrap();

        assert_eq!(outcome.metrics.bandwidth_3db_hz, None);
        assert_eq!(outcome.metrics.unity_gain_hz, None);
        assert_eq!(outcome.metrics.phase_margin_deg, None);
        assert_eq!(outcome.completion, AcCompletion::Complete);
        assert_eq!(calls, outcome.frequency_evaluations + 1);
        assert_eq!(outcome.samples.len(), outcome.frequency_evaluations);
    }

    #[test]
    fn adaptive_analysis_returns_the_first_crossing_before_gain_recovery() {
        let outcome =
            analyze_adaptive_ac(adaptive_config(), AdaptiveAcPolicy::COMPLETE, |frequency| {
                let gain_db = if frequency == 0.0 {
                    20.0
                } else {
                    let decade = frequency.log10();
                    if decade <= 3.0 {
                        20.0
                    } else if decade <= 4.0 {
                        20.0 - 10.0 * (decade - 3.0)
                    } else if decade <= 5.0 {
                        10.0 + 15.0 * (decade - 4.0)
                    } else {
                        25.0
                    }
                };
                Ok::<_, Infallible>(Complex64::new(10.0_f64.powf(gain_db / 20.0), 0.0))
            })
            .unwrap();

        let expected_first_crossing = 10.0_f64.powf(3.3);
        assert!(
            (outcome.metrics.bandwidth_3db_hz.unwrap() / expected_first_crossing - 1.0).abs()
                <= 0.005
        );
    }

    #[test]
    fn adaptive_analysis_rejects_non_finite_positive_frequency_response() {
        let error =
            analyze_adaptive_ac(adaptive_config(), AdaptiveAcPolicy::COMPLETE, |frequency| {
                Ok::<_, Infallible>(if frequency == 0.0 {
                    Complex64::new(10.0, 0.0)
                } else {
                    Complex64::new(f64::NAN, 0.0)
                })
            })
            .unwrap_err();

        assert!(matches!(
            error,
            AdaptiveAcError::NonFiniteResponse { frequency_hz: 1.0 }
        ));
    }
}
