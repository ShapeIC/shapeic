use std::error::Error;
use std::fmt;

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
        let mut assessment = TargetAssessment::default();
        assessment.check(
            TargetMetric::DcGainDb,
            self.min_dc_gain_db,
            Some(dc_gain_db),
        );
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
    use super::{AcMetrics, AnalysisMode, AnalysisTargets, TargetAssessment, TargetMetric};

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
}
