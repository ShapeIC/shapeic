//! Precomputed binding from candidate columns to numerical parameters.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use shapeic_lut::{MosCapacitanceMatrix, MosExtrinsicCapacitances};

use crate::compact_model::MosDeviceCapacitances;
use crate::macro_model::ResolvedPrimitiveBranch;
use crate::netlist::names::small_signal_param_name;

use super::candidate::{CandidateSchemaError, CandidateSet, candidate_column_indices};

const MOS_CAPACITANCE_PARAMETER_COUNT: usize =
    MosCapacitanceMatrix::INDEPENDENT_PARAMETERS.len() + MosExtrinsicCapacitances::PARAMETERS.len();

/// Resolves numerical parameters from one selected point per candidate set.
///
/// Column lookup and schema validation happen once in [`Self::new`]. The hot
/// path then uses only precomputed set and column offsets.
#[derive(Clone, Debug)]
pub struct CandidateParameterBinder<'a> {
    candidate_sets: Vec<&'a CandidateSet>,
    parameter_names: Vec<String>,
    sources: Vec<ParameterSource>,
}

impl<'a> CandidateParameterBinder<'a> {
    /// Precomputes the unique candidate column that supplies every parameter.
    pub fn new(
        parameter_order: &[String],
        candidate_sets: &[&'a CandidateSet],
    ) -> Result<Self, CandidateParameterBindingError> {
        let mut seen_parameters = HashSet::with_capacity(parameter_order.len());
        for parameter in parameter_order {
            if !seen_parameters.insert(parameter.as_str()) {
                return Err(CandidateParameterBindingError::DuplicateParameter {
                    parameter: parameter.clone(),
                });
            }
        }

        let column_indices = candidate_sets
            .iter()
            .map(|candidates| {
                candidate_column_indices(candidates).map_err(CandidateParameterBindingError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut sources = Vec::with_capacity(parameter_order.len());

        for parameter in parameter_order {
            let matches = column_indices
                .iter()
                .enumerate()
                .filter_map(|(set_index, columns)| {
                    columns
                        .get(parameter)
                        .copied()
                        .map(|column_index| ParameterSource {
                            set_index,
                            column_index,
                        })
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [] => {
                    return Err(CandidateParameterBindingError::MissingParameter {
                        parameter: parameter.clone(),
                    });
                }
                [source] => sources.push(*source),
                _ => {
                    return Err(CandidateParameterBindingError::AmbiguousParameter {
                        parameter: parameter.clone(),
                        candidate_sets: matches
                            .iter()
                            .map(|source| candidate_sets[source.set_index].name.clone())
                            .collect(),
                    });
                }
            }
        }

        Ok(Self {
            candidate_sets: candidate_sets.to_vec(),
            parameter_names: parameter_order.to_vec(),
            sources,
        })
    }

    /// Returns the numerical parameter names in binding order.
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Allocates and fills one parameter vector for the selected candidates.
    pub fn bind(
        &self,
        candidate_indices: &[usize],
    ) -> Result<Vec<f64>, CandidateParameterBindingError> {
        let mut values = vec![0.0; self.sources.len()];
        self.bind_into(candidate_indices, &mut values)?;
        Ok(values)
    }

    /// Fills a reusable parameter buffer without allocating.
    pub fn bind_into(
        &self,
        candidate_indices: &[usize],
        output: &mut [f64],
    ) -> Result<(), CandidateParameterBindingError> {
        if candidate_indices.len() != self.candidate_sets.len() {
            return Err(CandidateParameterBindingError::SelectionCountMismatch {
                expected: self.candidate_sets.len(),
                actual: candidate_indices.len(),
            });
        }
        if output.len() != self.sources.len() {
            return Err(CandidateParameterBindingError::OutputLengthMismatch {
                expected: self.sources.len(),
                actual: output.len(),
            });
        }
        for (candidates, point_index) in self.candidate_sets.iter().zip(candidate_indices) {
            if *point_index >= candidates.points.len() {
                return Err(CandidateParameterBindingError::CandidateIndexOutOfBounds {
                    candidate_set: candidates.name.clone(),
                    index: *point_index,
                    candidate_count: candidates.points.len(),
                });
            }
        }

        for (parameter_index, (parameter, source)) in
            self.parameter_names.iter().zip(&self.sources).enumerate()
        {
            let candidates = self.candidate_sets[source.set_index];
            let point_index = candidate_indices[source.set_index];
            let point = &candidates.points[point_index];
            let value = point.values[source.column_index].1;
            if !value.is_finite() {
                return Err(CandidateParameterBindingError::NonFiniteParameter {
                    parameter: parameter.clone(),
                    candidate_set: candidates.name.clone(),
                    candidate_index: point_index,
                });
            }
            output[parameter_index] = value;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct ParameterSource {
    set_index: usize,
    column_index: usize,
}

/// Errors produced while preparing or applying a candidate parameter binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateParameterBindingError {
    DuplicateParameter {
        parameter: String,
    },
    MissingParameter {
        parameter: String,
    },
    AmbiguousParameter {
        parameter: String,
        candidate_sets: Vec<String>,
    },
    EmptyCandidateSet {
        candidate_set: String,
    },
    DuplicateColumn {
        candidate_set: String,
        candidate_index: usize,
        column: String,
    },
    ColumnCountMismatch {
        candidate_set: String,
        candidate_index: usize,
        expected: usize,
        actual: usize,
    },
    ColumnNameMismatch {
        candidate_set: String,
        candidate_index: usize,
        column_index: usize,
        expected: String,
        actual: String,
    },
    SelectionCountMismatch {
        expected: usize,
        actual: usize,
    },
    OutputLengthMismatch {
        expected: usize,
        actual: usize,
    },
    CandidateIndexOutOfBounds {
        candidate_set: String,
        index: usize,
        candidate_count: usize,
    },
    NonFiniteParameter {
        parameter: String,
        candidate_set: String,
        candidate_index: usize,
    },
}

impl From<CandidateSchemaError> for CandidateParameterBindingError {
    fn from(error: CandidateSchemaError) -> Self {
        match error {
            CandidateSchemaError::EmptySet { set } => {
                Self::EmptyCandidateSet { candidate_set: set }
            }
            CandidateSchemaError::DuplicateColumn {
                set,
                point_index,
                column,
            } => Self::DuplicateColumn {
                candidate_set: set,
                candidate_index: point_index,
                column,
            },
            CandidateSchemaError::ColumnCountMismatch {
                set,
                point_index,
                expected,
                actual,
            } => Self::ColumnCountMismatch {
                candidate_set: set,
                candidate_index: point_index,
                expected,
                actual,
            },
            CandidateSchemaError::ColumnNameMismatch {
                set,
                point_index,
                column_index,
                expected,
                actual,
            } => Self::ColumnNameMismatch {
                candidate_set: set,
                candidate_index: point_index,
                column_index,
                expected,
                actual,
            },
        }
    }
}

impl fmt::Display for CandidateParameterBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateParameter { parameter } => {
                write!(
                    formatter,
                    "parameter order contains '{parameter}' more than once"
                )
            }
            Self::MissingParameter { parameter } => {
                write!(
                    formatter,
                    "no candidate set supplies parameter '{parameter}'"
                )
            }
            Self::AmbiguousParameter {
                parameter,
                candidate_sets,
            } => write!(
                formatter,
                "parameter '{parameter}' is supplied by multiple candidate sets: {}",
                candidate_sets.join(", ")
            ),
            Self::EmptyCandidateSet { candidate_set } => {
                write!(formatter, "candidate set '{candidate_set}' is empty")
            }
            Self::DuplicateColumn {
                candidate_set,
                candidate_index,
                column,
            } => write!(
                formatter,
                "candidate set '{candidate_set}' point {candidate_index} contains duplicate column '{column}'"
            ),
            Self::ColumnCountMismatch {
                candidate_set,
                candidate_index,
                expected,
                actual,
            } => write!(
                formatter,
                "candidate set '{candidate_set}' point {candidate_index} has {actual} columns, expected {expected}"
            ),
            Self::ColumnNameMismatch {
                candidate_set,
                candidate_index,
                column_index,
                expected,
                actual,
            } => write!(
                formatter,
                "candidate set '{candidate_set}' point {candidate_index} column {column_index} is '{actual}', expected '{expected}'"
            ),
            Self::SelectionCountMismatch { expected, actual } => write!(
                formatter,
                "parameter binding expected {expected} candidate indices, received {actual}"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                formatter,
                "parameter binding expected an output buffer of length {expected}, received {actual}"
            ),
            Self::CandidateIndexOutOfBounds {
                candidate_set,
                index,
                candidate_count,
            } => write!(
                formatter,
                "candidate index {index} is outside candidate set '{candidate_set}' with {candidate_count} points"
            ),
            Self::NonFiniteParameter {
                parameter,
                candidate_set,
                candidate_index,
            } => write!(
                formatter,
                "parameter '{parameter}' from candidate set '{candidate_set}' point {candidate_index} is not finite"
            ),
        }
    }
}

impl Error for CandidateParameterBindingError {}

/// Precomputed binding from candidate columns to per-branch MOS capacitances.
///
/// Branch topology and column lookup are resolved once during construction.
/// Bound capacitances follow [`Self::branches`] in the same deterministic order.
#[derive(Clone, Debug)]
pub struct CandidateCapacitanceBinder<'a> {
    branches: Vec<ResolvedPrimitiveBranch>,
    parameters: CandidateParameterBinder<'a>,
}

impl<'a> CandidateCapacitanceBinder<'a> {
    /// Resolves all intrinsic and extrinsic candidate columns for each branch.
    pub fn new(
        branches: &[ResolvedPrimitiveBranch],
        candidate_sets: &[&'a CandidateSet],
    ) -> Result<Self, CandidateCapacitanceBindingError> {
        let mut parameter_order =
            Vec::with_capacity(branches.len() * MOS_CAPACITANCE_PARAMETER_COUNT);
        for branch in branches {
            parameter_order.extend(
                MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
                    .into_iter()
                    .chain(MosExtrinsicCapacitances::PARAMETERS)
                    .map(|parameter| {
                        small_signal_param_name(
                            parameter,
                            branch.instance_path(),
                            branch.branch_name(),
                        )
                    }),
            );
        }
        let parameters = CandidateParameterBinder::new(&parameter_order, candidate_sets)?;
        Ok(Self {
            branches: branches.to_vec(),
            parameters,
        })
    }

    /// Returns the resolved branches in capacitance output order.
    pub fn branches(&self) -> &[ResolvedPrimitiveBranch] {
        &self.branches
    }

    /// Returns the number of scalar values required by [`Self::bind_into`].
    pub fn scratch_len(&self) -> usize {
        self.parameters.parameter_names().len()
    }

    /// Allocates and constructs one complete capacitance value per branch.
    pub fn bind(
        &self,
        candidate_indices: &[usize],
    ) -> Result<Vec<MosDeviceCapacitances>, CandidateCapacitanceBindingError> {
        let values = self.parameters.bind(candidate_indices)?;
        self.decode(&values).collect()
    }

    /// Updates reusable scalar and capacitance buffers without allocating.
    pub fn bind_into(
        &self,
        candidate_indices: &[usize],
        scratch: &mut [f64],
        output: &mut [MosDeviceCapacitances],
    ) -> Result<(), CandidateCapacitanceBindingError> {
        if scratch.len() != self.scratch_len() {
            return Err(CandidateCapacitanceBindingError::ScratchLengthMismatch {
                expected: self.scratch_len(),
                actual: scratch.len(),
            });
        }
        if output.len() != self.branches.len() {
            return Err(CandidateCapacitanceBindingError::OutputLengthMismatch {
                expected: self.branches.len(),
                actual: output.len(),
            });
        }
        self.parameters.bind_into(candidate_indices, scratch)?;
        for (destination, capacitances) in output.iter_mut().zip(self.decode(scratch)) {
            *destination = capacitances?;
        }
        Ok(())
    }

    fn decode<'b>(
        &'b self,
        values: &'b [f64],
    ) -> impl Iterator<Item = Result<MosDeviceCapacitances, CandidateCapacitanceBindingError>> + 'b
    {
        self.branches
            .iter()
            .zip(values.chunks_exact(MOS_CAPACITANCE_PARAMETER_COUNT))
            .map(|(branch, values)| {
                let extrinsic_offset = MosCapacitanceMatrix::INDEPENDENT_PARAMETERS.len();
                let extrinsic = MosExtrinsicCapacitances {
                    cgsol: values[extrinsic_offset],
                    cgdol: values[extrinsic_offset + 1],
                    cjs: values[extrinsic_offset + 2],
                    cjd: values[extrinsic_offset + 3],
                };
                MosDeviceCapacitances::from_total_parameters(&values[..extrinsic_offset], extrinsic)
                    .map_err(
                        |error| CandidateCapacitanceBindingError::InvalidCapacitances {
                            instance_path: branch.instance_path().to_owned(),
                            branch: branch.branch_name().to_owned(),
                            reason: error.to_string(),
                        },
                    )
            })
    }
}

/// Errors produced while preparing or applying a capacitance binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateCapacitanceBindingError {
    /// A required candidate column could not be bound or evaluated.
    Parameter(CandidateParameterBindingError),
    /// The reusable scalar buffer has the wrong length.
    ScratchLengthMismatch { expected: usize, actual: usize },
    /// The reusable capacitance buffer has the wrong length.
    OutputLengthMismatch { expected: usize, actual: usize },
    /// Bound values do not form valid complete MOS capacitances.
    InvalidCapacitances {
        instance_path: String,
        branch: String,
        reason: String,
    },
}

impl From<CandidateParameterBindingError> for CandidateCapacitanceBindingError {
    fn from(error: CandidateParameterBindingError) -> Self {
        Self::Parameter(error)
    }
}

impl fmt::Display for CandidateCapacitanceBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameter(error) => write!(formatter, "could not bind capacitances: {error}"),
            Self::ScratchLengthMismatch { expected, actual } => write!(
                formatter,
                "capacitance binding expected a scratch buffer of length {expected}, received {actual}"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                formatter,
                "capacitance binding expected an output buffer of length {expected}, received {actual}"
            ),
            Self::InvalidCapacitances {
                instance_path,
                branch,
                reason,
            } => write!(
                formatter,
                "candidate capacitances for '{instance_path}.{branch}' are invalid: {reason}"
            ),
        }
    }
}

impl Error for CandidateCapacitanceBindingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parameter(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CandidateCapacitanceBinder, CandidateCapacitanceBindingError, CandidateParameterBinder,
        CandidateParameterBindingError,
    };
    use crate::compact_model::MosDeviceCapacitances;
    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::macro_model::ResolvedPrimitiveBranch;
    use crate::netlist::names::small_signal_param_name;
    use shapeic_lut::{MosCapacitanceMatrix, MosExtrinsicCapacitances};

    fn point(values: &[(&str, f64)]) -> CandidatePoint {
        CandidatePoint::new(
            values
                .iter()
                .map(|(name, value)| ((*name).to_owned(), *value))
                .collect(),
        )
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn branch(instance: &str, branch: &str) -> ResolvedPrimitiveBranch {
        ResolvedPrimitiveBranch::new(
            instance.to_owned(),
            "mos_primitive".to_owned(),
            branch.to_owned(),
            format!("{instance}_G"),
            format!("{instance}_D"),
            format!("{instance}_S"),
            format!("{instance}_B"),
        )
    }

    fn capacitances(scale: f64) -> MosDeviceCapacitances {
        let intrinsic =
            [10.0, 2.0, 3.0, 1.0, 8.0, 2.0, 1.5, 2.5, 7.0].map(|value| value * scale * 1.0e-15);
        MosDeviceCapacitances::from_total_parameters(
            &intrinsic,
            MosExtrinsicCapacitances {
                cgsol: scale * 1.0e-15,
                cgdol: scale * 2.0e-15,
                cjs: scale * 3.0e-15,
                cjd: scale * 4.0e-15,
            },
        )
        .unwrap()
    }

    fn capacitance_point(instance: &str, branch: &str, scale: f64) -> CandidatePoint {
        let device = capacitances(scale);
        let intrinsic = device.intrinsic.to_ngspice_parameters();
        let independent = [
            intrinsic[0],
            intrinsic[1],
            intrinsic[2],
            intrinsic[4],
            intrinsic[5],
            intrinsic[6],
            intrinsic[8],
            intrinsic[9],
            intrinsic[10],
        ];
        let extrinsic = device.extrinsic;
        let values = MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
            .into_iter()
            .zip(independent)
            .chain(MosExtrinsicCapacitances::PARAMETERS.into_iter().zip([
                extrinsic.cgsol,
                extrinsic.cgdol,
                extrinsic.cjs,
                extrinsic.cjd,
            ]))
            .map(|(parameter, value)| (small_signal_param_name(parameter, instance, branch), value))
            .collect();
        CandidatePoint::new(values)
    }

    #[test]
    fn binds_multiple_candidate_sets_in_requested_parameter_order() {
        let diff_pair = CandidateSet::new(
            "xdp",
            vec![
                point(&[("gm__xdp__m1", 1.0), ("ro__xdp__m1", 10.0)]),
                point(&[("gm__xdp__m1", 2.0), ("ro__xdp__m1", 20.0)]),
            ],
        );
        let mirror = CandidateSet::new(
            "xcm",
            vec![point(&[("gm__xcm__m1", 3.0), ("ro__xcm__m1", 30.0)])],
        );
        let order = names(&["gm__xcm__m1", "ro__xdp__m1", "gm__xdp__m1"]);
        let binder = CandidateParameterBinder::new(&order, &[&diff_pair, &mirror]).unwrap();

        assert_eq!(binder.bind(&[1, 0]).unwrap(), [3.0, 20.0, 2.0]);
        assert_eq!(binder.parameter_names(), order);
    }

    #[test]
    fn binds_into_a_reusable_buffer_without_allocation() {
        let candidates = CandidateSet::new("stage", vec![point(&[("gm", 1.0), ("ro", 2.0)])]);
        let binder = CandidateParameterBinder::new(&names(&["ro", "gm"]), &[&candidates]).unwrap();
        let mut output = [0.0; 2];

        binder.bind_into(&[0], &mut output).unwrap();

        assert_eq!(output, [2.0, 1.0]);
    }

    #[test]
    fn rejects_missing_and_ambiguous_parameter_sources() {
        let left = CandidateSet::new("left", vec![point(&[("gm", 1.0)])]);
        let right = CandidateSet::new("right", vec![point(&[("gm", 2.0)])]);

        assert_eq!(
            CandidateParameterBinder::new(&names(&["ro"]), &[&left]).unwrap_err(),
            CandidateParameterBindingError::MissingParameter {
                parameter: "ro".to_owned(),
            }
        );
        assert_eq!(
            CandidateParameterBinder::new(&names(&["gm"]), &[&left, &right]).unwrap_err(),
            CandidateParameterBindingError::AmbiguousParameter {
                parameter: "gm".to_owned(),
                candidate_sets: vec!["left".to_owned(), "right".to_owned()],
            }
        );
    }

    #[test]
    fn rejects_invalid_selection_buffers_indices_and_values() {
        let candidates = CandidateSet::new("stage", vec![point(&[("gm", f64::NAN)])]);
        let binder = CandidateParameterBinder::new(&names(&["gm"]), &[&candidates]).unwrap();

        assert!(matches!(
            binder.bind(&[]),
            Err(CandidateParameterBindingError::SelectionCountMismatch { .. })
        ));
        assert!(matches!(
            binder.bind(&[1]),
            Err(CandidateParameterBindingError::CandidateIndexOutOfBounds { .. })
        ));
        assert!(matches!(
            binder.bind(&[0]),
            Err(CandidateParameterBindingError::NonFiniteParameter { .. })
        ));
        assert!(matches!(
            binder.bind_into(&[0], &mut []),
            Err(CandidateParameterBindingError::OutputLengthMismatch { .. })
        ));
    }

    #[test]
    fn binds_complete_capacitances_in_resolved_branch_order() {
        let diff_pair = CandidateSet::new(
            "xdp",
            vec![
                capacitance_point("xdp", "m1", 1.0),
                capacitance_point("xdp", "m1", 2.0),
            ],
        );
        let mirror = CandidateSet::new("xcm", vec![capacitance_point("xcm", "m2", 3.0)]);
        let branches = vec![branch("xdp", "m1"), branch("xcm", "m2")];
        let binder = CandidateCapacitanceBinder::new(&branches, &[&diff_pair, &mirror]).unwrap();

        assert_eq!(binder.branches(), branches);
        assert_eq!(
            binder.bind(&[1, 0]).unwrap(),
            [capacitances(2.0), capacitances(3.0)]
        );

        let mut scratch = vec![0.0; binder.scratch_len()];
        let mut output = binder.bind(&[1, 0]).unwrap();
        binder
            .bind_into(&[0, 0], &mut scratch, &mut output)
            .unwrap();
        assert_eq!(output, [capacitances(1.0), capacitances(3.0)]);
    }

    #[test]
    fn reports_missing_capacitance_columns_and_invalid_reusable_buffers() {
        let mut incomplete = capacitance_point("xdp", "m1", 1.0);
        incomplete.values.pop();
        let candidates = CandidateSet::new("xdp", vec![incomplete]);
        let branches = vec![branch("xdp", "m1")];

        assert!(matches!(
            CandidateCapacitanceBinder::new(&branches, &[&candidates]),
            Err(CandidateCapacitanceBindingError::Parameter(
                CandidateParameterBindingError::MissingParameter { parameter }
            )) if parameter == "cjd__xdp__m1"
        ));

        let complete = CandidateSet::new("xdp", vec![capacitance_point("xdp", "m1", 1.0)]);
        let binder = CandidateCapacitanceBinder::new(&branches, &[&complete]).unwrap();
        let mut scratch = vec![0.0; binder.scratch_len()];
        let mut output = binder.bind(&[0]).unwrap();
        assert!(matches!(
            binder.bind_into(&[0], &mut scratch[..1], &mut output),
            Err(CandidateCapacitanceBindingError::ScratchLengthMismatch { .. })
        ));
        assert!(matches!(
            binder.bind_into(&[0], &mut scratch, &mut []),
            Err(CandidateCapacitanceBindingError::OutputLengthMismatch { .. })
        ));
    }
}
