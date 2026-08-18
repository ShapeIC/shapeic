use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::iter::FusedIterator;

use super::table::ExplorationColumn;

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateSet {
    pub name: String,
    pub points: Vec<CandidatePoint>,
}
impl CandidateSet {
    pub fn new(name: impl Into<String>, points: Vec<CandidatePoint>) -> Self {
        Self {
            name: name.into(),
            points,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidatePoint {
    pub values: Vec<(String, f64)>,
}

impl CandidatePoint {
    pub fn new(values: Vec<(String, f64)>) -> Self {
        Self { values }
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.values
            .iter()
            .find_map(|(candidate_name, value)| (candidate_name == name).then_some(*value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CandidatePairSelection {
    pub left_index: usize,
    pub right_index: usize,
}

impl CandidatePairSelection {
    pub const fn new(left_index: usize, right_index: usize) -> Self {
        Self {
            left_index,
            right_index,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateJoinError {
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
    PairCountOverflow {
        left_count: usize,
        right_count: usize,
    },
}

impl fmt::Display for CandidateJoinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet { set } => {
                write!(formatter, "candidate set '{set}' is empty")
            }
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
            Self::PairCountOverflow {
                left_count,
                right_count,
            } => write!(
                formatter,
                "candidate pair count overflows usize: {left_count} * {right_count}"
            ),
        }
    }
}

impl Error for CandidateJoinError {}

impl From<CandidateSchemaError> for CandidateJoinError {
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

#[derive(Debug)]
pub struct CandidatePairJoin<'a> {
    _left: &'a CandidateSet,
    right: &'a CandidateSet,
    next_left: usize,
    next_right: usize,
    remaining: usize,
}

impl<'a> CandidatePairJoin<'a> {
    pub fn cartesian(
        left: &'a CandidateSet,
        right: &'a CandidateSet,
    ) -> Result<Self, CandidateJoinError> {
        candidate_column_indices(left)?;
        candidate_column_indices(right)?;
        let remaining = left.points.len().checked_mul(right.points.len()).ok_or(
            CandidateJoinError::PairCountOverflow {
                left_count: left.points.len(),
                right_count: right.points.len(),
            },
        )?;

        Ok(Self {
            _left: left,
            right,
            next_left: 0,
            next_right: 0,
            remaining,
        })
    }
}

impl Iterator for CandidatePairJoin<'_> {
    type Item = CandidatePairSelection;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let selection = CandidatePairSelection::new(self.next_left, self.next_right);
        self.remaining -= 1;
        self.next_right += 1;
        if self.next_right == self.right.points.len() {
            self.next_right = 0;
            self.next_left += 1;
        }
        Some(selection)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for CandidatePairJoin<'_> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl FusedIterator for CandidatePairJoin<'_> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CandidateSchemaError {
    EmptySet {
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
}

pub(crate) fn candidate_column_indices(
    candidates: &CandidateSet,
) -> Result<HashMap<String, usize>, CandidateSchemaError> {
    let Some(first) = candidates.points.first() else {
        return Err(CandidateSchemaError::EmptySet {
            set: candidates.name.clone(),
        });
    };

    let mut indices = HashMap::with_capacity(first.values.len());
    for (column_index, (column, _)) in first.values.iter().enumerate() {
        if indices.insert(column.clone(), column_index).is_some() {
            return Err(CandidateSchemaError::DuplicateColumn {
                set: candidates.name.clone(),
                point_index: 0,
                column: column.clone(),
            });
        }
    }

    for (point_index, point) in candidates.points.iter().enumerate().skip(1) {
        if point.values.len() != first.values.len() {
            return Err(CandidateSchemaError::ColumnCountMismatch {
                set: candidates.name.clone(),
                point_index,
                expected: first.values.len(),
                actual: point.values.len(),
            });
        }

        let mut point_columns = HashMap::with_capacity(point.values.len());
        for (column_index, ((expected, _), (actual, _))) in
            first.values.iter().zip(&point.values).enumerate()
        {
            if point_columns
                .insert(actual.as_str(), column_index)
                .is_some()
            {
                return Err(CandidateSchemaError::DuplicateColumn {
                    set: candidates.name.clone(),
                    point_index,
                    column: actual.clone(),
                });
            }
            if actual != expected {
                return Err(CandidateSchemaError::ColumnNameMismatch {
                    set: candidates.name.clone(),
                    point_index,
                    column_index,
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
        }
    }

    Ok(indices)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateSetBuildError {
    EmptyColumns,
    ColumnLengthMismatch {
        column: String,
        expected: usize,
        actual: usize,
    },
}
pub fn candidate_set_from_columns(
    name: impl Into<String>,
    columns: &[ExplorationColumn],
) -> Result<CandidateSet, CandidateSetBuildError> {
    let Some(first) = columns.first() else {
        return Err(CandidateSetBuildError::EmptyColumns);
    };

    let row_count = first.values.len();

    for column in columns {
        let actual = column.values.len();
        if actual != row_count {
            return Err(CandidateSetBuildError::ColumnLengthMismatch {
                column: column.name.clone(),
                expected: row_count,
                actual,
            });
        }
    }

    let mut points = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let values = columns
            .iter()
            .map(|column| (column.name.clone(), column.values[row_idx]))
            .collect();
        points.push(CandidatePoint::new(values));
    }

    let candidates = CandidateSet::new(name, points);
    debug_assert!(candidates.points.is_empty() || candidate_column_indices(&candidates).is_ok());
    Ok(candidates)
}

pub fn candidate_column_name(prefix: &str, column: &str) -> String {
    format!("{prefix}.{column}")
}

#[cfg(test)]
mod tests {
    use super::{
        CandidateJoinError, CandidatePairJoin, CandidatePairSelection, CandidatePoint,
        CandidateSchemaError, CandidateSet, candidate_column_indices,
    };

    fn point(values: &[(&str, f64)]) -> CandidatePoint {
        CandidatePoint::new(
            values
                .iter()
                .map(|(name, value)| ((*name).to_owned(), *value))
                .collect(),
        )
    }

    #[test]
    fn accesses_candidate_values_by_name() {
        let candidate = point(&[("width__xdp__m1", 4.0e-6), ("xdp.voutp", 1.0)]);

        assert_eq!(candidate.get("width__xdp__m1"), Some(4.0e-6));
        assert_eq!(candidate.get("xdp.voutp"), Some(1.0));
        assert_eq!(candidate.get("missing"), None);
    }

    #[test]
    fn preserves_original_pair_indices() {
        let selection = CandidatePairSelection::new(3, 7);

        assert_eq!(selection.left_index, 3);
        assert_eq!(selection.right_index, 7);
    }

    #[test]
    fn streams_the_cartesian_product_in_left_major_order() {
        let left = CandidateSet::new(
            "xdp",
            vec![point(&[("xdp.voutp", 0.8)]), point(&[("xdp.voutp", 1.0)])],
        );
        let right = CandidateSet::new(
            "xcm",
            vec![
                point(&[("xcm.voutp", 0.8)]),
                point(&[("xcm.voutp", 0.9)]),
                point(&[("xcm.voutp", 1.0)]),
            ],
        );

        let mut pairs = CandidatePairJoin::cartesian(&left, &right).unwrap();
        assert_eq!(pairs.len(), 6);
        assert_eq!(pairs.size_hint(), (6, Some(6)));
        assert_eq!(
            pairs.by_ref().collect::<Vec<_>>(),
            vec![
                CandidatePairSelection::new(0, 0),
                CandidatePairSelection::new(0, 1),
                CandidatePairSelection::new(0, 2),
                CandidatePairSelection::new(1, 0),
                CandidatePairSelection::new(1, 1),
                CandidatePairSelection::new(1, 2),
            ]
        );
        assert_eq!(pairs.len(), 0);
        assert_eq!(pairs.next(), None);
        assert_eq!(pairs.next(), None);
    }

    #[test]
    fn cartesian_selections_resolve_original_candidate_values() {
        let left = CandidateSet::new(
            "xdp",
            vec![point(&[("width__xdp__m1", 4.0e-6), ("xdp.voutp", 0.9)])],
        );
        let right = CandidateSet::new(
            "xcm",
            vec![point(&[("length__xcm__m1", 0.8e-6), ("xcm.voutp", 0.9)])],
        );
        let selection = CandidatePairJoin::cartesian(&left, &right)
            .unwrap()
            .next()
            .unwrap();

        assert_eq!(
            left.points[selection.left_index].get("width__xdp__m1"),
            Some(4.0e-6)
        );
        assert_eq!(
            right.points[selection.right_index].get("length__xcm__m1"),
            Some(0.8e-6)
        );
    }

    #[test]
    fn rejects_empty_candidate_sets_before_cartesian_iteration() {
        let empty = CandidateSet::new("xdp", Vec::new());
        let populated = CandidateSet::new("xcm", vec![point(&[("xcm.voutp", 1.0)])]);

        assert_eq!(
            CandidatePairJoin::cartesian(&empty, &populated).unwrap_err(),
            CandidateJoinError::EmptyCandidateSet {
                set: "xdp".to_owned(),
            }
        );
        assert_eq!(
            CandidatePairJoin::cartesian(&populated, &empty).unwrap_err(),
            CandidateJoinError::EmptyCandidateSet {
                set: "xdp".to_owned(),
            }
        );
    }

    #[test]
    fn reports_invalid_candidate_schemas_before_cartesian_iteration() {
        let invalid = CandidateSet::new(
            "xdp",
            vec![
                point(&[("xdp.voutp", 0.8), ("length__xdp__m1", 0.4e-6)]),
                point(&[("length__xdp__m1", 0.8e-6), ("xdp.voutp", 1.0)]),
            ],
        );
        let valid = CandidateSet::new("xcm", vec![point(&[("xcm.voutp", 1.0)])]);

        assert_eq!(
            CandidatePairJoin::cartesian(&invalid, &valid).unwrap_err(),
            CandidateJoinError::ColumnNameMismatch {
                set: "xdp".to_owned(),
                point_index: 1,
                column_index: 0,
                expected: "xdp.voutp".to_owned(),
                actual: "length__xdp__m1".to_owned(),
            }
        );
    }

    #[test]
    fn resolves_stable_candidate_column_indices() {
        let candidates = CandidateSet::new(
            "xdp",
            vec![
                point(&[("xdp.voutp", 0.8), ("length__xdp__m1", 0.4e-6)]),
                point(&[("xdp.voutp", 1.0), ("length__xdp__m1", 0.8e-6)]),
            ],
        );

        let indices = candidate_column_indices(&candidates).unwrap();

        assert_eq!(indices.get("xdp.voutp"), Some(&0));
        assert_eq!(indices.get("length__xdp__m1"), Some(&1));
        assert_eq!(indices.get("missing"), None);
    }

    #[test]
    fn rejects_candidate_points_with_inconsistent_column_order() {
        let candidates = CandidateSet::new(
            "xdp",
            vec![
                point(&[("xdp.voutp", 0.8), ("length__xdp__m1", 0.4e-6)]),
                point(&[("length__xdp__m1", 0.8e-6), ("xdp.voutp", 1.0)]),
            ],
        );

        assert_eq!(
            candidate_column_indices(&candidates),
            Err(CandidateSchemaError::ColumnNameMismatch {
                set: "xdp".to_owned(),
                point_index: 1,
                column_index: 0,
                expected: "xdp.voutp".to_owned(),
                actual: "length__xdp__m1".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_duplicate_candidate_columns() {
        let candidates = CandidateSet::new(
            "xcm",
            vec![point(&[("xcm.voutp", 0.8), ("xcm.voutp", 1.0)])],
        );

        assert_eq!(
            candidate_column_indices(&candidates),
            Err(CandidateSchemaError::DuplicateColumn {
                set: "xcm".to_owned(),
                point_index: 0,
                column: "xcm.voutp".to_owned(),
            })
        );
    }
}
