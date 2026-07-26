use std::error::Error;
use std::fmt;

use num_complex::Complex64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcMetrics {
    pub dc_gain_db: f64,
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
        validate_target(TargetMetric::DcGainDb, self.min_dc_gain_db, false)?;
        validate_target(
            TargetMetric::Bandwidth3DbHz,
            self.min_bandwidth_3db_hz,
            true,
        )?;
        validate_target(TargetMetric::UnityGainHz, self.min_unity_gain_hz, true)?;
        validate_target(
            TargetMetric::PhaseMarginDeg,
            self.min_phase_margin_deg,
            false,
        )
    }

    pub fn assess_dc_gain(self, dc_gain_db: f64) -> TargetAssessment {
        self.assess_metric(TargetMetric::DcGainDb, Some(dc_gain_db))
    }

    pub fn assess_metric(self, metric: TargetMetric, actual: Option<f64>) -> TargetAssessment {
        let minimum = match metric {
            TargetMetric::DcGainDb => self.min_dc_gain_db,
            TargetMetric::Bandwidth3DbHz => self.min_bandwidth_3db_hz,
            TargetMetric::UnityGainHz => self.min_unity_gain_hz,
            TargetMetric::PhaseMarginDeg => self.min_phase_margin_deg,
        };
        let mut assessment = TargetAssessment::default();
        assessment.check(metric, minimum, actual);
        assessment
    }

    pub fn assess_frequency_metrics(self, metrics: &AcMetrics) -> TargetAssessment {
        let mut assessment = TargetAssessment::default();
        assessment.check(
            TargetMetric::Bandwidth3DbHz,
            self.min_bandwidth_3db_hz,
            metrics.bandwidth_3db_hz,
        );
        assessment.check(
            TargetMetric::UnityGainHz,
            self.min_unity_gain_hz,
            metrics.unity_gain_hz,
        );
        assessment.check(
            TargetMetric::PhaseMarginDeg,
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
pub enum TargetMetric {
    DcGainDb,
    Bandwidth3DbHz,
    UnityGainHz,
    PhaseMarginDeg,
}

impl TargetMetric {
    pub const fn label(self) -> &'static str {
        match self {
            Self::DcGainDb => "GainDC",
            Self::Bandwidth3DbHz => "f3dB",
            Self::UnityGainHz => "UGF",
            Self::PhaseMarginDeg => "PM",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetFailure {
    pub metric: TargetMetric,
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

    fn check(&mut self, metric: TargetMetric, minimum: Option<f64>, actual: Option<f64>) {
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
}

impl AdaptiveAcPolicy {
    pub const COMPLETE: Self = Self {
        mode: AnalysisMode::FullInsight,
        targets: AnalysisTargets::NONE,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcCompletion {
    Complete,
    PrunedAfter(TargetMetric),
}

impl AcCompletion {
    pub const fn metric_evaluated(self, metric: TargetMetric) -> bool {
        let Some(stopped_after) = self.stopped_after() else {
            return true;
        };
        metric_order(metric) <= metric_order(stopped_after)
    }

    pub const fn stopped_after(self) -> Option<TargetMetric> {
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
    pub targets: TargetAssessment,
    pub completion: AcCompletion,
    pub samples: Vec<AdaptiveAcSample>,
    pub frequency_evaluations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidAdaptiveAcConfig;

impl fmt::Display for InvalidAdaptiveAcConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid adaptive AC configuration")
    }
}

impl Error for InvalidAdaptiveAcConfig {}

#[derive(Debug)]
pub enum AdaptiveAcError<E> {
    InvalidConfig(InvalidAdaptiveAcConfig),
    Evaluation { frequency_hz: f64, source: E },
    NonFiniteResponse { frequency_hz: f64 },
}

impl<E: fmt::Display> fmt::Display for AdaptiveAcError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => error.fmt(formatter),
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
    policy
        .targets
        .validate()
        .map_err(|_| AdaptiveAcError::InvalidConfig(InvalidAdaptiveAcConfig))?;

    let exact_dc = evaluate_loop_response(0.0).map_err(|source| AdaptiveAcError::Evaluation {
        frequency_hz: 0.0,
        source,
    })?;
    let mut sampler = AdaptiveSampler::new(config, &mut evaluate_loop_response);
    let low_frequency = sampler.sample(config.min_frequency_hz)?;
    let dc_response = if finite_response(exact_dc) {
        exact_dc
    } else {
        low_frequency.response
    };
    let dc_gain_db = magnitude_db(dc_response);
    let mut targets = policy
        .targets
        .assess_metric(TargetMetric::DcGainDb, Some(dc_gain_db));
    if should_prune(policy.mode, &targets) {
        return Ok(sampler.finish(
            AcMetrics {
                dc_gain_db,
                bandwidth_3db_hz: None,
                unity_gain_hz: None,
                phase_margin_deg: None,
            },
            targets,
            AcCompletion::PrunedAfter(TargetMetric::DcGainDb),
        ));
    }

    let bandwidth_3db_hz = sampler
        .first_downward_crossing(dc_gain_db - 3.0)?
        .map(|crossing| crossing.frequency_hz);
    let assessment = policy
        .targets
        .assess_metric(TargetMetric::Bandwidth3DbHz, bandwidth_3db_hz);
    let prune = should_prune(policy.mode, &assessment);
    targets.extend(assessment);
    if prune {
        return Ok(sampler.finish(
            AcMetrics {
                dc_gain_db,
                bandwidth_3db_hz,
                unity_gain_hz: None,
                phase_margin_deg: None,
            },
            targets,
            AcCompletion::PrunedAfter(TargetMetric::Bandwidth3DbHz),
        ));
    }

    let unity = sampler.first_downward_crossing(0.0)?;
    let unity_gain_hz = unity.map(|crossing| crossing.frequency_hz);
    let assessment = policy
        .targets
        .assess_metric(TargetMetric::UnityGainHz, unity_gain_hz);
    let prune = should_prune(policy.mode, &assessment);
    targets.extend(assessment);
    if prune {
        return Ok(sampler.finish(
            AcMetrics {
                dc_gain_db,
                bandwidth_3db_hz,
                unity_gain_hz,
                phase_margin_deg: None,
            },
            targets,
            AcCompletion::PrunedAfter(TargetMetric::UnityGainHz),
        ));
    }

    let phase_margin_deg = unity.map(|crossing| 180.0 + sampler.phase_at(crossing.frequency_hz));
    let assessment = policy
        .targets
        .assess_metric(TargetMetric::PhaseMarginDeg, phase_margin_deg);
    let prune = should_prune(policy.mode, &assessment);
    targets.extend(assessment);
    let completion = if prune {
        AcCompletion::PrunedAfter(TargetMetric::PhaseMarginDeg)
    } else {
        AcCompletion::Complete
    };

    Ok(sampler.finish(
        AcMetrics {
            dc_gain_db,
            bandwidth_3db_hz,
            unity_gain_hz,
            phase_margin_deg,
        },
        targets,
        completion,
    ))
}

fn should_prune(mode: AnalysisMode, assessment: &TargetAssessment) -> bool {
    mode == AnalysisMode::Prune && !assessment.passed()
}

const fn metric_order(metric: TargetMetric) -> u8 {
    match metric {
        TargetMetric::DcGainDb => 0,
        TargetMetric::Bandwidth3DbHz => 1,
        TargetMetric::UnityGainHz => 2,
        TargetMetric::PhaseMarginDeg => 3,
    }
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
    pub metric: TargetMetric,
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
                TargetMetric::Bandwidth3DbHz | TargetMetric::UnityGainHz
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
    metric: TargetMetric,
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

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use num_complex::Complex64;

    use super::{
        AcCompletion, AcMetrics, AdaptiveAcConfig, AdaptiveAcError, AdaptiveAcPolicy, AnalysisMode,
        AnalysisTargets, TargetAssessment, TargetMetric, analyze_adaptive_ac,
    };

    const TARGETS: AnalysisTargets = AnalysisTargets {
        min_dc_gain_db: Some(20.0),
        min_bandwidth_3db_hz: Some(100.0e6),
        min_unity_gain_hz: Some(1.0e9),
        min_phase_margin_deg: Some(45.0),
    };

    fn passing_metrics() -> AcMetrics {
        AcMetrics {
            dc_gain_db: 20.0,
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

    #[test]
    fn disabled_targets_always_pass() {
        let missing = AcMetrics {
            dc_gain_db: f64::NAN,
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
            dc_gain_db: 19.0,
            bandwidth_3db_hz: Some(90.0e6),
            unity_gain_hz: None,
            phase_margin_deg: Some(f64::NAN),
        };
        let assessment = TARGETS.assess_all(&metrics);

        assert_eq!(assessment.failures().len(), 4);
        assert_eq!(assessment.failures()[0].metric, TargetMetric::DcGainDb);
        assert_eq!(assessment.failures()[0].actual, Some(19.0));
        assert_eq!(
            assessment.failures()[1].metric,
            TargetMetric::Bandwidth3DbHz
        );
        assert_eq!(assessment.failures()[2].metric, TargetMetric::UnityGainHz);
        assert_eq!(assessment.failures()[2].actual, None);
        assert_eq!(
            assessment.failures()[3].metric,
            TargetMetric::PhaseMarginDeg
        );
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
            TargetMetric::UnityGainHz
        );

        let invalid_gain = AnalysisTargets {
            min_dc_gain_db: Some(f64::INFINITY),
            ..AnalysisTargets::NONE
        };
        assert_eq!(
            invalid_gain.validate().unwrap_err().metric,
            TargetMetric::DcGainDb
        );
    }

    #[test]
    fn prune_stops_on_failure_while_full_insight_continues() {
        let assessment = TargetAssessment {
            failures: vec![super::TargetFailure {
                metric: TargetMetric::DcGainDb,
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
        assert!((outcome.metrics.dc_gain_db - 20.0).abs() < 1.0e-12);
        assert!((outcome.metrics.bandwidth_3db_hz.unwrap() / pole_hz - 1.0).abs() <= 0.005);
        assert!((outcome.metrics.unity_gain_hz.unwrap() / expected_unity - 1.0).abs() <= 0.005);
        assert!((outcome.metrics.phase_margin_deg.unwrap() - expected_phase_margin).abs() <= 0.5);
        assert_eq!(outcome.completion, AcCompletion::Complete);
        assert!(outcome.frequency_evaluations < 80);
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
        };
        let outcome = analyze_adaptive_ac(adaptive_config(), policy, |frequency| {
            Ok::<_, Infallible>(one_pole_response(frequency, 10.0, 1.0e6))
        })
        .unwrap();

        assert_eq!(
            outcome.completion,
            AcCompletion::PrunedAfter(TargetMetric::Bandwidth3DbHz)
        );
        assert!(outcome.metrics.bandwidth_3db_hz.is_some());
        assert!(outcome.metrics.unity_gain_hz.is_none());
        assert!(
            outcome
                .completion
                .metric_evaluated(TargetMetric::Bandwidth3DbHz)
        );
        assert!(
            !outcome
                .completion
                .metric_evaluated(TargetMetric::UnityGainHz)
        );
        assert_eq!(outcome.targets.failures().len(), 1);
        assert_eq!(
            outcome.targets.failures()[0].metric,
            TargetMetric::Bandwidth3DbHz
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
