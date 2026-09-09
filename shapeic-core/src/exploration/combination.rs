//! Indexed compatibility traversal across arbitrary candidate sets.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use super::candidate::{
    CandidateSchemaError, CandidateSet, candidate_column_indices, canonical_join_value,
};

/// Equality between one column from each of two candidate sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateCombinationEquality {
    /// Index of the first candidate set in traversal order.
    pub left_set: usize,
    /// Column supplied by the first candidate set.
    pub left_column: String,
    /// Index of the second candidate set in traversal order.
    pub right_set: usize,
    /// Column supplied by the second candidate set.
    pub right_column: String,
}

impl CandidateCombinationEquality {
    /// Creates an equality between two candidate-set columns.
    pub fn new(
        left_set: usize,
        left_column: impl Into<String>,
        right_set: usize,
        right_column: impl Into<String>,
    ) -> Self {
        Self {
            left_set,
            left_column: left_column.into(),
            right_set,
            right_column: right_column.into(),
        }
    }
}

/// Indexed, allocation-free traversal of compatible multi-set selections.
///
/// For every constrained dimension, candidates are indexed by all equalities
/// connecting it to earlier dimensions. [`Self::next_selection`] performs a
/// depth-first traversal over matching buckets and reuses one selection slice.
#[derive(Debug)]
pub struct CandidateCombinationJoin<'a> {
    candidate_sets: Vec<&'a CandidateSet>,
    plans: Vec<DimensionPlan>,
    selection: Vec<usize>,
    next_options: Vec<usize>,
    depth: usize,
    yielded_empty_selection: bool,
    exhausted: bool,
}

impl<'a> CandidateCombinationJoin<'a> {
    /// Builds composite indexes for all cross-set equalities.
    pub fn new(
        candidate_sets: &[&'a CandidateSet],
        equalities: &[CandidateCombinationEquality],
    ) -> Result<Self, CandidateCombinationJoinError> {
        let columns = candidate_sets
            .iter()
            .map(|candidates| {
                candidate_column_indices(candidates).map_err(CandidateCombinationJoinError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut constraints = vec![Vec::<DimensionConstraint>::new(); candidate_sets.len()];

        for equality in equalities {
            validate_set_index(candidate_sets, equality.left_set)?;
            validate_set_index(candidate_sets, equality.right_set)?;
            if equality.left_set == equality.right_set {
                return Err(CandidateCombinationJoinError::SameSetEquality {
                    set_index: equality.left_set,
                    left_column: equality.left_column.clone(),
                    right_column: equality.right_column.clone(),
                });
            }
            let left_offset = column_offset(
                candidate_sets[equality.left_set],
                &columns[equality.left_set],
                &equality.left_column,
            )?;
            let right_offset = column_offset(
                candidate_sets[equality.right_set],
                &columns[equality.right_set],
                &equality.right_column,
            )?;
            validate_join_column(
                candidate_sets[equality.left_set],
                left_offset,
                &equality.left_column,
            )?;
            validate_join_column(
                candidate_sets[equality.right_set],
                right_offset,
                &equality.right_column,
            )?;

            let (dimension, constraint) = if equality.left_set < equality.right_set {
                (
                    equality.right_set,
                    DimensionConstraint {
                        prior_set: equality.left_set,
                        prior_offset: left_offset,
                        current_offset: right_offset,
                    },
                )
            } else {
                (
                    equality.left_set,
                    DimensionConstraint {
                        prior_set: equality.right_set,
                        prior_offset: right_offset,
                        current_offset: left_offset,
                    },
                )
            };
            constraints[dimension].push(constraint);
        }

        let plans = candidate_sets
            .iter()
            .zip(constraints)
            .map(|(candidates, constraints)| DimensionPlan::new(candidates, constraints))
            .collect();
        Ok(Self {
            candidate_sets: candidate_sets.to_vec(),
            plans,
            selection: vec![0; candidate_sets.len()],
            next_options: vec![0; candidate_sets.len()],
            depth: 0,
            yielded_empty_selection: false,
            exhausted: false,
        })
    }

    /// Returns the next compatible reusable selection slice.
    pub fn next_selection(&mut self) -> Option<&[usize]> {
        if self.exhausted {
            return None;
        }
        if self.candidate_sets.is_empty() {
            if self.yielded_empty_selection {
                self.exhausted = true;
                return None;
            }
            self.yielded_empty_selection = true;
            return Some(&self.selection);
        }

        loop {
            if self.depth == self.candidate_sets.len() {
                self.depth -= 1;
                return Some(&self.selection);
            }

            let option = self.next_options[self.depth];
            if let Some(candidate_index) = self.candidate_at(self.depth, option) {
                self.next_options[self.depth] += 1;
                self.selection[self.depth] = candidate_index;
                self.depth += 1;
                if self.depth < self.next_options.len() {
                    self.next_options[self.depth] = 0;
                }
            } else if self.depth == 0 {
                self.exhausted = true;
                return None;
            } else {
                self.next_options[self.depth] = 0;
                self.depth -= 1;
            }
        }
    }

    /// Returns the number of candidate-set indices in each selection.
    pub fn selection_len(&self) -> usize {
        self.selection.len()
    }

    fn candidate_at(&mut self, dimension: usize, option: usize) -> Option<usize> {
        let selection = &self.selection;
        let candidate_sets = &self.candidate_sets;
        let plan = &mut self.plans[dimension];
        let Some(index) = plan.index.as_ref() else {
            return (option < plan.candidate_count).then_some(option);
        };

        plan.probe_key.clear();
        for constraint in &plan.constraints {
            let point =
                &candidate_sets[constraint.prior_set].points[selection[constraint.prior_set]];
            let value = point.values[constraint.prior_offset].1;
            plan.probe_key.push(
                canonical_join_value(value)
                    .expect("join values were validated during construction"),
            );
        }
        index
            .get(plan.probe_key.as_slice())
            .and_then(|matches| matches.get(option))
            .copied()
    }
}

#[derive(Debug)]
struct DimensionPlan {
    candidate_count: usize,
    constraints: Vec<DimensionConstraint>,
    index: Option<HashMap<Box<[u64]>, Vec<usize>>>,
    probe_key: Vec<u64>,
}

impl DimensionPlan {
    fn new(candidates: &CandidateSet, constraints: Vec<DimensionConstraint>) -> Self {
        if constraints.is_empty() {
            return Self {
                candidate_count: candidates.points.len(),
                constraints,
                index: None,
                probe_key: Vec::new(),
            };
        }

        let mut index = HashMap::<Box<[u64]>, Vec<usize>>::new();
        for (candidate_index, point) in candidates.points.iter().enumerate() {
            let key = constraints
                .iter()
                .map(|constraint| {
                    canonical_join_value(point.values[constraint.current_offset].1)
                        .expect("join values were validated during construction")
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            index.entry(key).or_default().push(candidate_index);
        }
        Self {
            candidate_count: candidates.points.len(),
            probe_key: Vec::with_capacity(constraints.len()),
            constraints,
            index: Some(index),
        }
    }
}

#[derive(Clone, Debug)]
struct DimensionConstraint {
    prior_set: usize,
    prior_offset: usize,
    current_offset: usize,
}

/// Errors produced while preparing indexed multi-set compatibility traversal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateCombinationJoinError {
    EmptyCandidateSet {
        set: String,
    },
    DuplicateColumn {
        set: String,
        point_index: usize,
        column: String,
    },
    ColumnCountMismatch {
        set: String,
        point_index: usize,
        expected: usize,
        actual: usize,
    },
    ColumnNameMismatch {
        set: String,
        point_index: usize,
        column_index: usize,
        expected: String,
        actual: String,
    },
    CandidateSetIndexOutOfBounds {
        set_index: usize,
        set_count: usize,
    },
    SameSetEquality {
        set_index: usize,
        left_column: String,
        right_column: String,
    },
    MissingColumn {
        set: String,
        column: String,
    },
    NonFiniteJoinValue {
        set: String,
        point_index: usize,
        column: String,
    },
}

impl From<CandidateSchemaError> for CandidateCombinationJoinError {
    fn from(error: CandidateSchemaError) -> Self {
        match error {
            CandidateSchemaError::EmptySet { set } => Self::EmptyCandidateSet { set },
            CandidateSchemaError::DuplicateColumn {
                set,
                point_index,
                column,
            } => Self::DuplicateColumn {
                set,
                point_index,
                column,
            },
            CandidateSchemaError::ColumnCountMismatch {
                set,
                point_index,
                expected,
                actual,
            } => Self::ColumnCountMismatch {
                set,
                point_index,
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
                set,
                point_index,
                column_index,
                expected,
                actual,
            },
        }
    }
}

impl fmt::Display for CandidateCombinationJoinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet { set } => write!(formatter, "candidate set '{set}' is empty"),
            Self::DuplicateColumn {
                set,
                point_index,
                column,
            } => write!(
                formatter,
                "candidate set '{set}' point {point_index} contains duplicate column '{column}'"
            ),
            Self::ColumnCountMismatch {
                set,
                point_index,
                expected,
                actual,
            } => write!(
                formatter,
                "candidate set '{set}' point {point_index} has {actual} columns, expected {expected}"
            ),
            Self::ColumnNameMismatch {
                set,
                point_index,
                column_index,
                expected,
                actual,
            } => write!(
                formatter,
                "candidate set '{set}' point {point_index} column {column_index} is '{actual}', expected '{expected}'"
            ),
            Self::CandidateSetIndexOutOfBounds {
                set_index,
                set_count,
            } => write!(
                formatter,
                "candidate set index {set_index} is outside the {set_count} supplied sets"
            ),
            Self::SameSetEquality {
                set_index,
                left_column,
                right_column,
            } => write!(
                formatter,
                "candidate equality cannot compare '{left_column}' and '{right_column}' within set {set_index}"
            ),
            Self::MissingColumn { set, column } => {
                write!(formatter, "candidate set '{set}' has no column '{column}'")
            }
            Self::NonFiniteJoinValue {
                set,
                point_index,
                column,
            } => write!(
                formatter,
                "candidate set '{set}' point {point_index} has a non-finite join value in '{column}'"
            ),
        }
    }
}

impl Error for CandidateCombinationJoinError {}

fn validate_set_index(
    candidate_sets: &[&CandidateSet],
    set_index: usize,
) -> Result<(), CandidateCombinationJoinError> {
    if set_index >= candidate_sets.len() {
        return Err(
            CandidateCombinationJoinError::CandidateSetIndexOutOfBounds {
                set_index,
                set_count: candidate_sets.len(),
            },
        );
    }
    Ok(())
}

fn column_offset(
    candidates: &CandidateSet,
    columns: &HashMap<String, usize>,
    column: &str,
) -> Result<usize, CandidateCombinationJoinError> {
    columns
        .get(column)
        .copied()
        .ok_or_else(|| CandidateCombinationJoinError::MissingColumn {
            set: candidates.name.clone(),
            column: column.to_owned(),
        })
}

fn validate_join_column(
    candidates: &CandidateSet,
    offset: usize,
    column: &str,
) -> Result<(), CandidateCombinationJoinError> {
    for (point_index, point) in candidates.points.iter().enumerate() {
        if canonical_join_value(point.values[offset].1).is_none() {
            return Err(CandidateCombinationJoinError::NonFiniteJoinValue {
                set: candidates.name.clone(),
                point_index,
                column: column.to_owned(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CandidateCombinationEquality, CandidateCombinationJoin, CandidateCombinationJoinError,
    };
    use crate::exploration::candidate::{CandidatePoint, CandidateSet};

    fn point(values: &[(&str, f64)]) -> CandidatePoint {
        CandidatePoint::new(
            values
                .iter()
                .map(|(name, value)| ((*name).to_owned(), *value))
                .collect(),
        )
    }

    #[test]
    fn traverses_only_compatible_selections_across_three_sets() {
        let first = CandidateSet::new(
            "first",
            vec![point(&[("first.x", 0.0)]), point(&[("first.x", 1.0)])],
        );
        let second = CandidateSet::new(
            "second",
            vec![
                point(&[("second.x", 0.0), ("second.y", 10.0)]),
                point(&[("second.x", 0.0), ("second.y", 20.0)]),
                point(&[("second.x", 1.0), ("second.y", 10.0)]),
            ],
        );
        let third = CandidateSet::new(
            "third",
            vec![point(&[("third.y", 10.0)]), point(&[("third.y", 20.0)])],
        );
        let equalities = [
            CandidateCombinationEquality::new(0, "first.x", 1, "second.x"),
            CandidateCombinationEquality::new(1, "second.y", 2, "third.y"),
        ];
        let mut join =
            CandidateCombinationJoin::new(&[&first, &second, &third], &equalities).unwrap();

        assert_eq!(join.selection_len(), 3);
        let first_pointer = join.next_selection().unwrap().as_ptr();
        let mut selections = vec![vec![0, 0, 0]];
        while let Some(selection) = join.next_selection() {
            assert_eq!(selection.as_ptr(), first_pointer);
            selections.push(selection.to_vec());
        }
        assert_eq!(
            selections,
            vec![vec![0, 0, 0], vec![0, 1, 1], vec![1, 2, 0]]
        );
        assert_eq!(join.next_selection(), None);
    }

    #[test]
    fn falls_back_to_cartesian_dimensions_and_supports_zero_sets() {
        let first = CandidateSet::new("first", vec![point(&[("a", 0.0)]), point(&[("a", 1.0)])]);
        let second = CandidateSet::new("second", vec![point(&[("b", 0.0)]), point(&[("b", 1.0)])]);
        let mut join = CandidateCombinationJoin::new(&[&first, &second], &[]).unwrap();
        let mut selections = Vec::new();
        while let Some(selection) = join.next_selection() {
            selections.push(selection.to_vec());
        }
        assert_eq!(
            selections,
            vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
        );

        let mut empty = CandidateCombinationJoin::new(&[], &[]).unwrap();
        assert_eq!(empty.next_selection(), Some([].as_slice()));
        assert_eq!(empty.next_selection(), None);
    }

    #[test]
    fn validates_equality_set_indices_columns_and_values() {
        let first = CandidateSet::new("first", vec![point(&[("x", f64::NAN)])]);
        let second = CandidateSet::new("second", vec![point(&[("y", 1.0)])]);

        assert!(matches!(
            CandidateCombinationJoin::new(
                &[&first, &second],
                &[CandidateCombinationEquality::new(0, "x", 2, "y")]
            ),
            Err(
                CandidateCombinationJoinError::CandidateSetIndexOutOfBounds {
                    set_index: 2,
                    set_count: 2,
                }
            )
        ));
        assert!(matches!(
            CandidateCombinationJoin::new(
                &[&first, &second],
                &[CandidateCombinationEquality::new(0, "missing", 1, "y")]
            ),
            Err(CandidateCombinationJoinError::MissingColumn { set, column })
                if set == "first" && column == "missing"
        ));
        assert!(matches!(
            CandidateCombinationJoin::new(
                &[&first, &second],
                &[CandidateCombinationEquality::new(0, "x", 1, "y")]
            ),
            Err(CandidateCombinationJoinError::NonFiniteJoinValue {
                set,
                point_index: 0,
                column,
            }) if set == "first" && column == "x"
        ));
    }
}
