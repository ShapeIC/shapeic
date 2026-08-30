//! Minimal netlist-first small-signal testbench execution.
//!
//! A testbench is supplied as SPICE source text or as a `.spice` file. The
//! symbolic MNA is compiled once and reused to instantiate one numerical
//! testbench per candidate. Numerical compact or layout-aware models may be
//! stamped before running an AC sweep or a single DC node-voltage solution.

use std::error::Error;
use std::fmt;
use std::path::Path;

use num_complex::Complex64;
use shapeic_mna::mna::{MnaError, MnaResult, mna_from_spice, mna_from_spice_file};
use shapeic_mna::numeric::{NumericMnaError, NumericMnaSystem, PreparedNumericMna};

use crate::analysis::{
    AdaptiveAcConfig, AdaptiveAcError, AdaptiveAcOutcome, AdaptiveAcPolicy, analyze_adaptive_ac,
};

/// Sign applied to a node-to-node voltage transfer before AC metric extraction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransferPolarity {
    /// Evaluates `V(output_node) / V(input_node)`.
    #[default]
    Positive,
    /// Evaluates `-V(output_node) / V(input_node)`.
    Negative,
}

impl TransferPolarity {
    fn apply(self, response: Complex64) -> Complex64 {
        match self {
            Self::Positive => response,
            Self::Negative => -response,
        }
    }
}

/// A single-ended node-to-node voltage transfer function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferFunction {
    /// Input voltage node, measured relative to ground.
    pub input_node: String,
    /// Output voltage node, measured relative to ground.
    pub output_node: String,
    /// Sign applied to the solved node transfer before AC metric extraction.
    pub polarity: TransferPolarity,
}

impl TransferFunction {
    /// Creates the transfer function `V(output_node) / V(input_node)`.
    pub fn new(input_node: impl Into<String>, output_node: impl Into<String>) -> Self {
        Self {
            input_node: input_node.into(),
            output_node: output_node.into(),
            polarity: TransferPolarity::Positive,
        }
    }

    /// Selects the sign applied to `V(output_node) / V(input_node)`.
    pub fn with_polarity(mut self, polarity: TransferPolarity) -> Self {
        self.polarity = polarity;
        self
    }
}

/// AC analysis attached to a SPICE testbench.
#[derive(Clone, Debug, PartialEq)]
pub struct AcAnalysis {
    /// Transfer function evaluated at every requested frequency.
    pub transfer_function: TransferFunction,
    /// Adaptive frequency search configuration.
    pub config: AdaptiveAcConfig,
    /// Selected metrics, targets, and pruning mode.
    pub policy: AdaptiveAcPolicy,
}

/// A DC analysis that resolves one node voltage relative to ground.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DcNodeVoltageAnalysis {
    node: String,
}

impl DcNodeVoltageAnalysis {
    /// Creates an analysis for `V(node)` at `s = 0`.
    pub fn new(node: impl Into<String>) -> Self {
        Self { node: node.into() }
    }

    /// Returns the node whose voltage is resolved relative to ground.
    pub fn node(&self) -> &str {
        &self.node
    }
}

/// Result of resolving one signed DC node voltage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DcNodeVoltageOutcome {
    voltage_v: f64,
}

impl DcNodeVoltageOutcome {
    /// Returns the signed node voltage in volts.
    pub const fn voltage_v(self) -> f64 {
        self.voltage_v
    }
}

#[derive(Clone, Debug)]
struct PreparedLinearTestbench {
    mna: PreparedNumericMna,
}

impl PreparedLinearTestbench {
    fn from_mna(system: &MnaResult, parameter_order: &[&str]) -> Result<Self, NumericMnaError> {
        Ok(Self {
            mna: PreparedNumericMna::new(system, parameter_order)?,
        })
    }

    fn parameter_names(&self) -> &[String] {
        self.mna.parameter_names()
    }

    fn instantiate(
        &mut self,
        parameter_values: &[f64],
    ) -> Result<NumericMnaSystem, NumericMnaError> {
        self.mna.instantiate(parameter_values)
    }
}

impl AcAnalysis {
    /// Creates an AC analysis for one transfer function.
    pub const fn new(
        transfer_function: TransferFunction,
        config: AdaptiveAcConfig,
        policy: AdaptiveAcPolicy,
    ) -> Self {
        Self {
            transfer_function,
            config,
            policy,
        }
    }
}

/// A compiled AC testbench reusable across candidate parameter values.
#[derive(Clone, Debug)]
pub struct PreparedAcTestbench {
    prepared: PreparedLinearTestbench,
    analysis: AcAnalysis,
}

impl PreparedAcTestbench {
    /// Builds and compiles a testbench directly from SPICE source text.
    ///
    /// `parameter_order` defines the exact order expected by
    /// [`Self::instantiate`]. The symbolic MNA is generated entirely in memory.
    pub fn from_spice(
        source: &str,
        parameter_order: &[&str],
        analysis: AcAnalysis,
    ) -> Result<Self, AcTestbenchBuildError> {
        let system = mna_from_spice(source).map_err(AcTestbenchBuildError::Mna)?;
        Self::from_mna(&system, parameter_order, analysis)
    }

    /// Builds and compiles a testbench from one `.spice` file.
    ///
    /// The file is read once during preparation and no intermediate `.cir` file
    /// is written.
    pub fn from_spice_file(
        path: &Path,
        parameter_order: &[&str],
        analysis: AcAnalysis,
    ) -> Result<Self, AcTestbenchBuildError> {
        let system = mna_from_spice_file(path).map_err(AcTestbenchBuildError::Mna)?;
        Self::from_mna(&system, parameter_order, analysis)
    }

    /// Returns the numerical parameter names in their required order.
    pub fn parameter_names(&self) -> &[String] {
        self.prepared.parameter_names()
    }

    /// Returns the AC analysis attached to this testbench.
    pub const fn analysis(&self) -> &AcAnalysis {
        &self.analysis
    }

    /// Instantiates the compiled MNA for one candidate.
    ///
    /// The returned testbench exposes its numerical MNA through
    /// [`AcTestbench::system_mut`] so candidate-specific compact or physical
    /// models can be stamped before analysis.
    pub fn instantiate(
        &mut self,
        parameter_values: &[f64],
    ) -> Result<AcTestbench, NumericMnaError> {
        Ok(AcTestbench {
            system: self.prepared.instantiate(parameter_values)?,
            analysis: self.analysis.clone(),
        })
    }

    fn from_mna(
        system: &MnaResult,
        parameter_order: &[&str],
        analysis: AcAnalysis,
    ) -> Result<Self, AcTestbenchBuildError> {
        let prepared = PreparedLinearTestbench::from_mna(system, parameter_order)
            .map_err(AcTestbenchBuildError::NumericMna)?;
        Ok(Self { prepared, analysis })
    }
}

/// A compiled DC node-voltage testbench reusable across candidate values.
#[derive(Clone, Debug)]
pub struct PreparedDcNodeVoltageTestbench {
    prepared: PreparedLinearTestbench,
    analysis: DcNodeVoltageAnalysis,
}

impl PreparedDcNodeVoltageTestbench {
    /// Builds and compiles a testbench directly from SPICE source text.
    pub fn from_spice(
        source: &str,
        parameter_order: &[&str],
        analysis: DcNodeVoltageAnalysis,
    ) -> Result<Self, DcNodeVoltageTestbenchBuildError> {
        let system = mna_from_spice(source).map_err(DcNodeVoltageTestbenchBuildError::Mna)?;
        Self::from_mna(&system, parameter_order, analysis)
    }

    /// Builds and compiles a testbench from one `.spice` file.
    pub fn from_spice_file(
        path: &Path,
        parameter_order: &[&str],
        analysis: DcNodeVoltageAnalysis,
    ) -> Result<Self, DcNodeVoltageTestbenchBuildError> {
        let system = mna_from_spice_file(path).map_err(DcNodeVoltageTestbenchBuildError::Mna)?;
        Self::from_mna(&system, parameter_order, analysis)
    }

    /// Returns the numerical parameter names in their required order.
    pub fn parameter_names(&self) -> &[String] {
        self.prepared.parameter_names()
    }

    /// Returns the DC node-voltage analysis attached to this testbench.
    pub const fn analysis(&self) -> &DcNodeVoltageAnalysis {
        &self.analysis
    }

    /// Instantiates the compiled MNA for one candidate.
    pub fn instantiate(
        &mut self,
        parameter_values: &[f64],
    ) -> Result<DcNodeVoltageTestbench, NumericMnaError> {
        Ok(DcNodeVoltageTestbench {
            system: self.prepared.instantiate(parameter_values)?,
            analysis: self.analysis.clone(),
        })
    }

    fn from_mna(
        system: &MnaResult,
        parameter_order: &[&str],
        analysis: DcNodeVoltageAnalysis,
    ) -> Result<Self, DcNodeVoltageTestbenchBuildError> {
        let prepared = PreparedLinearTestbench::from_mna(system, parameter_order)
            .map_err(DcNodeVoltageTestbenchBuildError::NumericMna)?;
        Ok(Self { prepared, analysis })
    }
}

/// A numerical AC testbench instantiated for one candidate.
#[derive(Clone, Debug)]
pub struct AcTestbench {
    system: NumericMnaSystem,
    analysis: AcAnalysis,
}

impl AcTestbench {
    /// Returns the instantiated numerical MNA system.
    pub const fn system(&self) -> &NumericMnaSystem {
        &self.system
    }

    /// Returns the instantiated numerical MNA system for additional stamping.
    pub const fn system_mut(&mut self) -> &mut NumericMnaSystem {
        &mut self.system
    }

    /// Returns the AC analysis attached to this testbench.
    pub const fn analysis(&self) -> &AcAnalysis {
        &self.analysis
    }

    /// Executes the selected adaptive AC metrics for the configured transfer.
    pub fn analyze(
        &self,
    ) -> Result<AdaptiveAcOutcome, AdaptiveAcError<AcTestbenchEvaluationError>> {
        let transfer = &self.analysis.transfer_function;
        analyze_adaptive_ac(
            self.analysis.config,
            self.analysis.policy,
            |frequency_hz| match self
                .system
                .solve_transfer(frequency_hz, &transfer.input_node, &transfer.output_node)
                .map_err(AcTestbenchEvaluationError::NumericMna)?
            {
                Some(response) => Ok(transfer.polarity.apply(response)),
                None if frequency_hz == 0.0 => Ok(Complex64::new(f64::NAN, f64::NAN)),
                None => Err(AcTestbenchEvaluationError::SingularSystem),
            },
        )
    }
}

/// A numerical DC node-voltage testbench instantiated for one candidate.
#[derive(Clone, Debug)]
pub struct DcNodeVoltageTestbench {
    system: NumericMnaSystem,
    analysis: DcNodeVoltageAnalysis,
}

impl DcNodeVoltageTestbench {
    /// Returns the instantiated numerical MNA system.
    pub const fn system(&self) -> &NumericMnaSystem {
        &self.system
    }

    /// Returns the instantiated numerical MNA system for additional stamping.
    pub const fn system_mut(&mut self) -> &mut NumericMnaSystem {
        &mut self.system
    }

    /// Returns the DC node-voltage analysis attached to this testbench.
    pub const fn analysis(&self) -> &DcNodeVoltageAnalysis {
        &self.analysis
    }

    /// Solves the MNA once at `s = 0` and returns the signed node voltage.
    pub fn analyze(&self) -> Result<DcNodeVoltageOutcome, DcNodeVoltageTestbenchEvaluationError> {
        let node = self.analysis.node();
        let voltage = self
            .system
            .solve_node(0.0, node)
            .map_err(DcNodeVoltageTestbenchEvaluationError::NumericMna)?
            .ok_or(DcNodeVoltageTestbenchEvaluationError::SingularSystem)?;
        let imaginary_tolerance = 1.0e-12 * voltage.re.abs().max(1.0);
        if voltage.im.abs() > imaginary_tolerance {
            return Err(DcNodeVoltageTestbenchEvaluationError::NonRealVoltage {
                node: node.to_owned(),
                imaginary_v: voltage.im,
            });
        }
        Ok(DcNodeVoltageOutcome {
            voltage_v: voltage.re,
        })
    }
}

/// Errors produced while building and compiling an AC testbench.
#[derive(Debug)]
pub enum AcTestbenchBuildError {
    /// SPICE preprocessing or symbolic MNA construction failed.
    Mna(MnaError),
    /// Numerical MNA preparation failed.
    NumericMna(NumericMnaError),
}

impl fmt::Display for AcTestbenchBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mna(error) => write!(formatter, "could not build testbench MNA: {error:?}"),
            Self::NumericMna(error) => {
                write!(
                    formatter,
                    "could not prepare numerical testbench MNA: {error}"
                )
            }
        }
    }
}

impl Error for AcTestbenchBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Mna(_) => None,
            Self::NumericMna(error) => Some(error),
        }
    }
}

/// Errors produced while building and compiling a DC node-voltage testbench.
#[derive(Debug)]
pub enum DcNodeVoltageTestbenchBuildError {
    /// SPICE preprocessing or symbolic MNA construction failed.
    Mna(MnaError),
    /// Numerical MNA preparation failed.
    NumericMna(NumericMnaError),
}

impl fmt::Display for DcNodeVoltageTestbenchBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mna(error) => write!(formatter, "could not build testbench MNA: {error:?}"),
            Self::NumericMna(error) => {
                write!(
                    formatter,
                    "could not prepare numerical testbench MNA: {error}"
                )
            }
        }
    }
}

impl Error for DcNodeVoltageTestbenchBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Mna(_) => None,
            Self::NumericMna(error) => Some(error),
        }
    }
}

/// Errors produced while evaluating an instantiated AC testbench.
#[derive(Clone, Debug, PartialEq)]
pub enum AcTestbenchEvaluationError {
    /// Numerical MNA solution failed.
    NumericMna(NumericMnaError),
    /// The numerical MNA matrix was singular at a swept frequency.
    SingularSystem,
}

impl fmt::Display for AcTestbenchEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NumericMna(error) => error.fmt(formatter),
            Self::SingularSystem => formatter.write_str("numerical testbench MNA is singular"),
        }
    }
}

impl Error for AcTestbenchEvaluationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NumericMna(error) => Some(error),
            Self::SingularSystem => None,
        }
    }
}

/// Errors produced while resolving a DC node voltage.
#[derive(Clone, Debug, PartialEq)]
pub enum DcNodeVoltageTestbenchEvaluationError {
    /// Numerical MNA solution failed.
    NumericMna(NumericMnaError),
    /// The numerical MNA matrix is singular at `s = 0`.
    SingularSystem,
    /// A DC solution unexpectedly contained a significant imaginary component.
    NonRealVoltage {
        /// Node whose DC voltage was requested.
        node: String,
        /// Imaginary component returned by the numerical solver.
        imaginary_v: f64,
    },
}

impl fmt::Display for DcNodeVoltageTestbenchEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NumericMna(error) => error.fmt(formatter),
            Self::SingularSystem => {
                formatter.write_str("numerical testbench MNA is singular at s = 0")
            }
            Self::NonRealVoltage { node, imaginary_v } => write!(
                formatter,
                "DC voltage at node '{node}' has imaginary component {imaginary_v} V"
            ),
        }
    }
}

impl Error for DcNodeVoltageTestbenchEvaluationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NumericMna(error) => Some(error),
            Self::SingularSystem | Self::NonRealVoltage { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::convert::Infallible;
    use std::f64::consts::PI;
    use std::path::Path;

    use ndarray::array;
    use num_complex::Complex64;

    use super::{
        AcAnalysis, DcNodeVoltageAnalysis, DcNodeVoltageTestbenchEvaluationError,
        PreparedAcTestbench, PreparedDcNodeVoltageTestbench, TransferFunction, TransferPolarity,
    };
    use crate::analysis::{
        AcMetric, AcMetricSet, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets,
        analyze_adaptive_ac,
    };
    use shapeic_mna::numeric::NumericMnaError;

    const CONFIG: AdaptiveAcConfig = AdaptiveAcConfig {
        min_frequency_hz: 1.0,
        max_frequency_hz: 1.0e6,
        coarse_points_per_decade: 4,
        crossing_relative_tolerance: 1.0e-4,
        max_refinement_steps: 32,
        retain_samples: false,
    };

    fn analysis(metrics: AcMetricSet) -> AcAnalysis {
        AcAnalysis::new(
            TransferFunction::new("VIN", "VOUT"),
            CONFIG,
            AdaptiveAcPolicy {
                mode: AnalysisMode::FullInsight,
                targets: AnalysisTargets::NONE,
                metrics,
            },
        )
    }

    #[test]
    fn applies_transfer_polarity_without_changing_magnitude() {
        let response = Complex64::new(3.0, -4.0);
        let positive = TransferPolarity::Positive.apply(response);
        let negative = TransferPolarity::Negative.apply(response);

        assert_eq!(positive, response);
        assert_eq!(negative, -response);
        assert_eq!(positive.norm(), negative.norm());
        assert!(((negative / positive).arg().abs() - PI).abs() < 1.0e-12);
    }

    #[test]
    fn negative_polarity_normalizes_an_inverting_one_pole_loop_response() {
        let dc_gain = 10.0;
        let pole_hz = 1.0e3;
        let outcome = analyze_adaptive_ac(
            CONFIG,
            AdaptiveAcPolicy {
                mode: AnalysisMode::FullInsight,
                targets: AnalysisTargets::NONE,
                metrics: AcMetricSet::ALL,
            },
            |frequency_hz| {
                let loop_response = dc_gain / Complex64::new(1.0, frequency_hz / pole_hz);
                let inverting_amplifier_response = -loop_response;
                Ok::<_, Infallible>(TransferPolarity::Negative.apply(inverting_amplifier_response))
            },
        )
        .unwrap();

        let expected_phase_margin = 180.0 - (99.0_f64.sqrt()).atan().to_degrees();
        assert!((outcome.metrics.dc_gain_db.unwrap() - 20.0).abs() < 1.0e-12);
        assert!((outcome.metrics.phase_margin_deg.unwrap() - expected_phase_margin).abs() < 0.5);
    }

    #[test]
    fn prepares_a_file_once_and_analyzes_stamped_parameterized_candidates() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rc_ac_testbench.spice");
        let metrics = AcMetricSet::from_metric(AcMetric::DcGainDb).with(AcMetric::Bandwidth3DbHz);
        let mut prepared =
            PreparedAcTestbench::from_spice_file(&path, &["r", "c"], analysis(metrics))
                .expect("RC testbench should prepare");

        assert_eq!(prepared.parameter_names(), ["r", "c"]);
        assert_eq!(
            prepared.analysis().transfer_function,
            TransferFunction::new("VIN", "VOUT")
        );

        let first = prepared
            .instantiate(&[1.0e3, 1.0e-6])
            .expect("first candidate should instantiate")
            .analyze()
            .expect("first candidate should analyze");
        let mut second_testbench = prepared
            .instantiate(&[2.0e3, 1.0e-6])
            .expect("second candidate should instantiate");
        assert_eq!(second_testbench.system().base_matrix().nrows(), 3);
        second_testbench
            .system_mut()
            .stamp_port_admittance(
                &["OUT".to_owned()],
                &BTreeMap::from([("OUT".to_owned(), "VOUT".to_owned())]),
                array![[0.0]].view(),
                array![[0.0]].view(),
            )
            .expect("candidate-specific stamp should be accepted");
        let second = second_testbench
            .analyze()
            .expect("second candidate should analyze");

        assert!(first.metrics.dc_gain_db.unwrap().abs() < 1.0e-12);
        assert!(second.metrics.dc_gain_db.unwrap().abs() < 1.0e-12);
        let three_db_ratio = (10.0_f64.powf(3.0 / 10.0) - 1.0).sqrt();
        let first_expected = three_db_ratio / (2.0 * PI * 1.0e3 * 1.0e-6);
        let second_expected = three_db_ratio / (2.0 * PI * 2.0e3 * 1.0e-6);
        let first_actual = first.metrics.bandwidth_3db_hz.unwrap();
        let second_actual = second.metrics.bandwidth_3db_hz.unwrap();
        assert!(
            (first_actual / first_expected - 1.0).abs() < 1.0e-3,
            "expected {first_expected}, got {first_actual}"
        );
        assert!(
            (second_actual / second_expected - 1.0).abs() < 1.0e-3,
            "expected {second_expected}, got {second_actual}"
        );
    }

    #[test]
    fn resolves_a_signed_parameterized_node_voltage_once_at_dc() {
        let source = "\
V1 VIN 0 -2
R1 VIN VOUT r
R2 VOUT 0 r
C1 VOUT 0 c
";
        let mut prepared = PreparedDcNodeVoltageTestbench::from_spice(
            source,
            &["r", "c"],
            DcNodeVoltageAnalysis::new("VOUT"),
        )
        .expect("DC testbench should prepare");

        assert_eq!(prepared.parameter_names(), ["r", "c"]);
        assert_eq!(prepared.analysis().node(), "VOUT");

        let first = prepared
            .instantiate(&[1.0e3, 1.0e-12])
            .expect("first candidate should instantiate")
            .analyze()
            .expect("first candidate should solve");
        let second = prepared
            .instantiate(&[1.0e3, 1.0])
            .expect("second candidate should instantiate")
            .analyze()
            .expect("second candidate should solve");

        assert!((first.voltage_v() + 1.0).abs() < 1.0e-12);
        assert_eq!(first, second, "capacitance must not affect the s = 0 solve");
    }

    #[test]
    fn reports_a_missing_dc_node() {
        let mut prepared = PreparedDcNodeVoltageTestbench::from_spice(
            "V1 VIN 0 1\n",
            &[],
            DcNodeVoltageAnalysis::new("MISSING"),
        )
        .expect("DC testbench should prepare");
        let error = prepared
            .instantiate(&[])
            .expect("testbench should instantiate")
            .analyze()
            .expect_err("missing node should fail");

        assert_eq!(
            error,
            DcNodeVoltageTestbenchEvaluationError::NumericMna(NumericMnaError::MissingNode(
                "MISSING".to_owned()
            ))
        );
    }

    #[test]
    fn reports_a_singular_dc_system() {
        let source = "\
V1 VIN 0 1
C1 VFLOAT 0 1e-12
";
        let mut prepared = PreparedDcNodeVoltageTestbench::from_spice(
            source,
            &[],
            DcNodeVoltageAnalysis::new("VFLOAT"),
        )
        .expect("DC testbench should prepare");
        let error = prepared
            .instantiate(&[])
            .expect("testbench should instantiate")
            .analyze()
            .expect_err("floating DC node should make the system singular");

        assert_eq!(error, DcNodeVoltageTestbenchEvaluationError::SingularSystem);
    }
}
