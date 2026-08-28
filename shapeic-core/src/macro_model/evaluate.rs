//! Candidate binding and numerical instantiation for macro AC testbenches.

use std::error::Error;
use std::fmt;

use shapeic_lut::{MosCapacitanceMatrix, MosExtrinsicCapacitances};
use shapeic_layout::PhysicalLookupTable;
use shapeic_mna::numeric::NumericMnaError;

use crate::analysis::{AdaptiveAcError, AdaptiveAcOutcome};
use crate::compact_model::{
    MosDeviceCapacitances, ResolvedCapacitanceStampError, stamp_resolved_mos_capacitances,
};
use crate::exploration::binding::{
    CandidateCapacitanceBinder, CandidateCapacitanceBindingError, CandidateParameterBinder,
    CandidateParameterBindingError,
};
use crate::exploration::candidate::CandidateSet;
use crate::testbench::{AcTestbench, AcTestbenchEvaluationError};

use super::{
    CandidatePhysicalBinder, CandidatePhysicalBindingError, MacroAnalysisDomain,
    PhysicalCandidateStampOutcome, PreparedMacroAcTestbench,
};

const ZERO_CAPACITANCES: MosDeviceCapacitances = MosDeviceCapacitances {
    intrinsic: MosCapacitanceMatrix {
        values: [[0.0; 4]; 4],
    },
    extrinsic: MosExtrinsicCapacitances {
        cgsol: 0.0,
        cgdol: 0.0,
        cjs: 0.0,
        cjd: 0.0,
    },
};

/// Reusable candidate-to-MNA instantiation path for one macro AC testbench.
///
/// Parameter and capacitance columns are resolved during construction. Each
/// subsequent call to [`Self::instantiate`] uses preallocated buffers, then
/// applies the resolved MOS capacitance stamps before returning the testbench.
#[derive(Debug)]
pub struct PreparedMacroAcCandidateEvaluator<'a> {
    testbench: PreparedMacroAcTestbench,
    parameters: CandidateParameterBinder<'a>,
    capacitances: CandidateCapacitanceBinder<'a>,
    parameter_values: Vec<f64>,
    capacitance_scratch: Vec<f64>,
    capacitance_values: Vec<MosDeviceCapacitances>,
    physical: Option<CandidatePhysicalBinder<'a>>,
}

impl<'a> PreparedMacroAcCandidateEvaluator<'a> {
    /// Precomputes all candidate-column bindings for the prepared testbench.
    pub fn new(
        testbench: PreparedMacroAcTestbench,
        candidate_sets: &[&'a CandidateSet],
    ) -> Result<Self, PreparedMacroAcCandidateEvaluatorError> {
        Self::new_with_physical_lut(testbench, candidate_sets, None)
    }

    /// Precomputes candidate bindings and attaches a shared physical LUT when
    /// the prepared testbench is layout-aware.
    pub fn new_with_physical_lut(
        testbench: PreparedMacroAcTestbench,
        candidate_sets: &[&'a CandidateSet],
        physical_lut: Option<&'a PhysicalLookupTable>,
    ) -> Result<Self, PreparedMacroAcCandidateEvaluatorError> {
        let parameters =
            CandidateParameterBinder::new(testbench.parameter_names(), candidate_sets)?;
        let capacitances =
            CandidateCapacitanceBinder::new(testbench.primitive_branches(), candidate_sets)?;
        let physical = match (testbench.domain(), physical_lut) {
            (MacroAnalysisDomain::Electrical, _) => None,
            (MacroAnalysisDomain::LayoutAware, Some(lut)) => Some(CandidatePhysicalBinder::new(
                testbench.physical_primitives(),
                candidate_sets,
                lut,
            )?),
            (MacroAnalysisDomain::LayoutAware, None) => {
                return Err(PreparedMacroAcCandidateEvaluatorError::MissingPhysicalLut);
            }
        };
        let parameter_values = vec![0.0; parameters.parameter_names().len()];
        let capacitance_scratch = vec![0.0; capacitances.scratch_len()];
        let capacitance_values = vec![ZERO_CAPACITANCES; capacitances.branches().len()];
        Ok(Self {
            testbench,
            parameters,
            capacitances,
            parameter_values,
            capacitance_scratch,
            capacitance_values,
            physical,
        })
    }

    /// Returns the underlying prepared macro testbench.
    pub const fn testbench(&self) -> &PreparedMacroAcTestbench {
        &self.testbench
    }

    /// Creates a worker-private evaluator for an electrical testbench.
    ///
    /// The compiled testbench and immutable bindings are cloned, while all hot
    /// path scratch buffers are allocated independently for the worker.
    pub(crate) fn fork_electrical(&self) -> Self {
        assert_eq!(self.testbench.domain(), MacroAnalysisDomain::Electrical);
        Self {
            testbench: self.testbench.clone(),
            parameters: self.parameters.clone(),
            capacitances: self.capacitances.clone(),
            parameter_values: vec![0.0; self.parameter_values.len()],
            capacitance_scratch: vec![0.0; self.capacitance_scratch.len()],
            capacitance_values: vec![ZERO_CAPACITANCES; self.capacitance_values.len()],
            physical: None,
        }
    }

    /// Binds, instantiates, and capacitance-stamps one candidate selection.
    pub fn instantiate(
        &mut self,
        candidate_indices: &[usize],
    ) -> Result<AcTestbench, PreparedMacroAcCandidateEvaluatorError> {
        self.parameters
            .bind_into(candidate_indices, &mut self.parameter_values)?;
        self.capacitances.bind_into(
            candidate_indices,
            &mut self.capacitance_scratch,
            &mut self.capacitance_values,
        )?;
        let mut candidate = self.testbench.instantiate(&self.parameter_values)?;
        stamp_resolved_mos_capacitances(
            candidate.system_mut(),
            self.capacitances.branches(),
            &self.capacitance_values,
        )?;
        if let Some(physical) = &mut self.physical
            && physical.stamp(candidate.system_mut(), candidate_indices)?
                == PhysicalCandidateStampOutcome::OutOfDomain
        {
            return Err(PreparedMacroAcCandidateEvaluatorError::PhysicalDomainRejected);
        }
        Ok(candidate)
    }

    /// Instantiates and evaluates one selection while classifying physical
    /// out-of-domain points as a normal candidate rejection.
    pub fn evaluate(
        &mut self,
        candidate_indices: &[usize],
    ) -> Result<MacroAcCandidateEvaluation, MacroAcCandidateAnalysisError> {
        match self.analyze(candidate_indices) {
            Ok(outcome) => Ok(MacroAcCandidateEvaluation::Outcome(outcome)),
            Err(MacroAcCandidateAnalysisError::Candidate(
                PreparedMacroAcCandidateEvaluatorError::PhysicalDomainRejected,
            )) => Ok(MacroAcCandidateEvaluation::PhysicalDomainRejected),
            Err(error) => Err(error),
        }
    }

    /// Instantiates and executes the configured AC analysis for one selection.
    pub fn analyze(
        &mut self,
        candidate_indices: &[usize],
    ) -> Result<AdaptiveAcOutcome, MacroAcCandidateAnalysisError> {
        let candidate = self
            .instantiate(candidate_indices)
            .map_err(MacroAcCandidateAnalysisError::Candidate)?;
        candidate
            .analyze()
            .map_err(MacroAcCandidateAnalysisError::Analysis)
    }
}

/// Classified result of evaluating one macro AC candidate.
#[derive(Debug, PartialEq)]
pub enum MacroAcCandidateEvaluation {
    Outcome(AdaptiveAcOutcome),
    PhysicalDomainRejected,
}

/// Errors produced while instantiating or analyzing one macro candidate.
#[derive(Debug)]
pub enum MacroAcCandidateAnalysisError {
    /// Candidate values could not be bound, instantiated, or stamped.
    Candidate(PreparedMacroAcCandidateEvaluatorError),
    /// AC metric extraction failed after successful candidate instantiation.
    Analysis(AdaptiveAcError<AcTestbenchEvaluationError>),
}

impl fmt::Display for MacroAcCandidateAnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Candidate(error) => error.fmt(formatter),
            Self::Analysis(error) => error.fmt(formatter),
        }
    }
}

impl Error for MacroAcCandidateAnalysisError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Candidate(error) => Some(error),
            Self::Analysis(error) => Some(error),
        }
    }
}

/// Errors produced while preparing or instantiating macro AC candidates.
#[derive(Debug)]
pub enum PreparedMacroAcCandidateEvaluatorError {
    /// Numerical MNA parameters could not be bound from the candidate sets.
    ParameterBinding(CandidateParameterBindingError),
    /// MOS capacitances could not be bound from the candidate sets.
    CapacitanceBinding(CandidateCapacitanceBindingError),
    /// The prepared numerical MNA could not be instantiated.
    Instantiate(NumericMnaError),
    /// Resolved MOS capacitances could not be stamped into the candidate MNA.
    CapacitanceStamp(ResolvedCapacitanceStampError),
    /// A layout-aware testbench was prepared without a physical LUT.
    MissingPhysicalLut,
    /// The selected local geometry or bias lies outside the physical LUT axes.
    PhysicalDomainRejected,
    /// Physical candidate columns, LUT data, or MNA stamps were invalid.
    Physical(CandidatePhysicalBindingError),
}

impl From<CandidateParameterBindingError> for PreparedMacroAcCandidateEvaluatorError {
    fn from(error: CandidateParameterBindingError) -> Self {
        Self::ParameterBinding(error)
    }
}

impl From<CandidateCapacitanceBindingError> for PreparedMacroAcCandidateEvaluatorError {
    fn from(error: CandidateCapacitanceBindingError) -> Self {
        Self::CapacitanceBinding(error)
    }
}

impl From<NumericMnaError> for PreparedMacroAcCandidateEvaluatorError {
    fn from(error: NumericMnaError) -> Self {
        Self::Instantiate(error)
    }
}

impl From<ResolvedCapacitanceStampError> for PreparedMacroAcCandidateEvaluatorError {
    fn from(error: ResolvedCapacitanceStampError) -> Self {
        Self::CapacitanceStamp(error)
    }
}

impl From<CandidatePhysicalBindingError> for PreparedMacroAcCandidateEvaluatorError {
    fn from(error: CandidatePhysicalBindingError) -> Self {
        Self::Physical(error)
    }
}

impl fmt::Display for PreparedMacroAcCandidateEvaluatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParameterBinding(error) => {
                write!(
                    formatter,
                    "could not bind candidate MNA parameters: {error}"
                )
            }
            Self::CapacitanceBinding(error) => {
                write!(formatter, "could not bind candidate capacitances: {error}")
            }
            Self::Instantiate(error) => {
                write!(formatter, "could not instantiate candidate MNA: {error}")
            }
            Self::CapacitanceStamp(error) => error.fmt(formatter),
            Self::MissingPhysicalLut => {
                formatter.write_str("layout-aware analysis requires a registered physical LUT")
            }
            Self::PhysicalDomainRejected => {
                formatter.write_str("candidate lies outside the physical LUT domain")
            }
            Self::Physical(error) => error.fmt(formatter),
        }
    }
}

impl Error for PreparedMacroAcCandidateEvaluatorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ParameterBinding(error) => Some(error),
            Self::CapacitanceBinding(error) => Some(error),
            Self::Instantiate(error) => Some(error),
            Self::CapacitanceStamp(error) => Some(error),
            Self::Physical(error) => Some(error),
            Self::MissingPhysicalLut | Self::PhysicalDomainRejected => None,
        }
    }
}
