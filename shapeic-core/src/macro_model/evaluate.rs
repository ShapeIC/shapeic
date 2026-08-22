//! Candidate binding and numerical instantiation for macro AC testbenches.

use std::error::Error;
use std::fmt;

use shapeic_lut::{MosCapacitanceMatrix, MosExtrinsicCapacitances};
use shapeic_mna::numeric::NumericMnaError;

use crate::compact_model::{
    MosDeviceCapacitances, ResolvedCapacitanceStampError, stamp_resolved_mos_capacitances,
};
use crate::exploration::binding::{
    CandidateCapacitanceBinder, CandidateCapacitanceBindingError, CandidateParameterBinder,
    CandidateParameterBindingError,
};
use crate::exploration::candidate::CandidateSet;
use crate::testbench::AcTestbench;

use super::PreparedMacroAcTestbench;

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
}

impl<'a> PreparedMacroAcCandidateEvaluator<'a> {
    /// Precomputes all candidate-column bindings for the prepared testbench.
    pub fn new(
        testbench: PreparedMacroAcTestbench,
        candidate_sets: &[&'a CandidateSet],
    ) -> Result<Self, PreparedMacroAcCandidateEvaluatorError> {
        let parameters =
            CandidateParameterBinder::new(testbench.parameter_names(), candidate_sets)?;
        let capacitances =
            CandidateCapacitanceBinder::new(testbench.primitive_branches(), candidate_sets)?;
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
        })
    }

    /// Returns the underlying prepared macro testbench.
    pub const fn testbench(&self) -> &PreparedMacroAcTestbench {
        &self.testbench
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
        Ok(candidate)
    }
}

/// Errors produced while preparing or instantiating macro AC candidates.
#[derive(Clone, Debug, PartialEq)]
pub enum PreparedMacroAcCandidateEvaluatorError {
    /// Numerical MNA parameters could not be bound from the candidate sets.
    ParameterBinding(CandidateParameterBindingError),
    /// MOS capacitances could not be bound from the candidate sets.
    CapacitanceBinding(CandidateCapacitanceBindingError),
    /// The prepared numerical MNA could not be instantiated.
    Instantiate(NumericMnaError),
    /// Resolved MOS capacitances could not be stamped into the candidate MNA.
    CapacitanceStamp(ResolvedCapacitanceStampError),
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
        }
    }
}
