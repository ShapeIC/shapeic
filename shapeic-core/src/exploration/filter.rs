use std::error::Error;
use std::fmt;

use super::candidate::{CandidateSchemaError, CandidateSet, candidate_column_indices};

/// An inclusive numeric range applied to one candidate column.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateFilter {
    column: String,
    minimum: Option<f64>,
    maximum: Option<f64>,
}

impl CandidateFilter {
    /// Creates an inclusive candidate-column range.
    pub fn new(
        column: impl Into<String>,
        minimum: Option<f64>,
        maximum: Option<f64>,
    ) -> Result<Self, InvalidCandidateFilter> {
        let filter = Self {
            column: column.into(),
            minimum,
            maximum,
        };
        filter.validate()?;
        Ok(filter)
    }

    /// Creates the inclusive condition `value >= minimum`.
    pub fn at_least(
        column: impl Into<String>,
        minimum: f64,
    ) -> Result<Self, InvalidCandidateFilter> {
        Self::new(column, Some(minimum), None)
    }

    /// Creates the inclusive condition `value <= maximum`.
    pub fn at_most(
        column: impl Into<String>,
        maximum: f64,
    ) -> Result<Self, InvalidCandidateFilter> {
        Self::new(column, None, Some(maximum))
    }

    /// Creates the inclusive condition `minimum <= value <= maximum`.
    pub fn between(
        column: impl Into<String>,
        minimum: f64,
        maximum: f64,
    ) -> Result<Self, InvalidCandidateFilter> {
        Self::new(column, Some(minimum), Some(maximum))
    }

    /// Returns the candidate column evaluated by this filter.
    pub fn column(&self) -> &str {
        &self.column
    }

    /// Returns the inclusive lower bound, when present.
    pub const fn minimum(&self) -> Option<f64> {
        self.minimum
    }

    /// Returns the inclusive upper bound, when present.
    pub const fn maximum(&self) -> Option<f64> {
        self.maximum
    }

    /// Returns whether a finite value is inside this filter's inclusive range.
    ///
    /// NaN and infinite values are never accepted.
    pub fn accepts_value(&self, value: f64) -> bool {
        value.is_finite()
            && self.minimum.is_none_or(|minimum| value >= minimum)
            && self.maximum.is_none_or(|maximum| value <= maximum)
    }

    fn validate(&self) -> Result<(), InvalidCandidateFilter> {
        if self.column.is_empty() {
            return Err(InvalidCandidateFilter::EmptyColumn);
        }
        if self.minimum.is_none() && self.maximum.is_none() {
            return Err(InvalidCandidateFilter::MissingBounds {
                column: self.column.clone(),
            });
        }
        if self.minimum.is_some_and(|minimum| !minimum.is_finite()) {
            return Err(InvalidCandidateFilter::NonFiniteMinimum {
                column: self.column.clone(),
            });
        }
        if self.maximum.is_some_and(|maximum| !maximum.is_finite()) {
            return Err(InvalidCandidateFilter::NonFiniteMaximum {
                column: self.column.clone(),
            });
        }
        if self
            .minimum
            .zip(self.maximum)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return Err(InvalidCandidateFilter::InvertedRange {
                column: self.column.clone(),
            });
        }
        Ok(())
    }
}

/// An invalid declarative candidate filter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidCandidateFilter {
    /// The target column name is empty.
    EmptyColumn,
    /// Neither a lower nor an upper bound was provided.
    MissingBounds { column: String },
    /// The lower bound is NaN or infinite.
    NonFiniteMinimum { column: String },
    /// The upper bound is NaN or infinite.
    NonFiniteMaximum { column: String },
    /// The lower bound is greater than the upper bound.
    InvertedRange { column: String },
}

impl fmt::Display for InvalidCandidateFilter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyColumn => write!(formatter, "candidate filter column cannot be empty"),
            Self::MissingBounds { column } => {
                write!(formatter, "candidate filter for '{column}' has no bounds")
            }
            Self::NonFiniteMinimum { column } => write!(
                formatter,
                "candidate filter for '{column}' has a non-finite minimum"
            ),
            Self::NonFiniteMaximum { column } => write!(
                formatter,
                "candidate filter for '{column}' has a non-finite maximum"
            ),
            Self::InvertedRange { column } => write!(
                formatter,
                "candidate filter for '{column}' has a minimum greater than its maximum"
            ),
        }
    }
}

impl Error for InvalidCandidateFilter {}

/// Counts produced by one in-place candidate filtering operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateFilterReport {
    input_count: usize,
    retained_count: usize,
}

impl CandidateFilterReport {
    /// Returns the number of candidates before filtering.
    pub const fn input_count(self) -> usize {
        self.input_count
    }

    /// Returns the number of candidates retained after applying every filter.
    pub const fn retained_count(self) -> usize {
        self.retained_count
    }

    /// Returns the number of candidates rejected by the filters.
    pub const fn rejected_count(self) -> usize {
        self.input_count - self.retained_count
    }
}

/// Errors produced while filtering a candidate set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateFilterError {
    /// A candidate contains the same column more than once.
    DuplicateColumn {
        set: String,
        point_index: usize,
        column: String,
    },
    /// A candidate has a different number of columns from the first candidate.
    ColumnCountMismatch {
        set: String,
        point_index: usize,
        expected: usize,
        actual: usize,
    },
    /// A candidate column does not match the position established by the first candidate.
    ColumnNameMismatch {
        set: String,
        point_index: usize,
        column_index: usize,
        expected: String,
        actual: String,
    },
    /// A filter references a column that is not present in the candidate set.
    MissingColumn { set: String, column: String },
    /// A filtered column contains NaN or an infinite value.
    NonFiniteValue {
        set: String,
        point_index: usize,
        column: String,
    },
}

impl fmt::Display for CandidateFilterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::MissingColumn { set, column } => {
                write!(formatter, "candidate set '{set}' has no column '{column}'")
            }
            Self::NonFiniteValue {
                set,
                point_index,
                column,
            } => write!(
                formatter,
                "candidate set '{set}' point {point_index} has a non-finite filter value in '{column}'"
            ),
        }
    }
}

impl Error for CandidateFilterError {}

/// Retains candidates that satisfy every filter without cloning the candidate set.
///
/// Filters are combined with logical AND. The candidate schema, referenced
/// columns, and filtered values are validated before the set is modified, so an
/// error leaves `candidates` unchanged. Empty sets and empty filter lists are
/// valid no-op operations.
pub fn retain_candidate_set(
    candidates: &mut CandidateSet,
    filters: &[CandidateFilter],
) -> Result<CandidateFilterReport, CandidateFilterError> {
    retain_candidate_set_impl(candidates, filters, None)
}

pub(crate) fn retain_candidate_set_with_indices(
    candidates: &mut CandidateSet,
    filters: &[CandidateFilter],
) -> Result<(CandidateFilterReport, Vec<usize>), CandidateFilterError> {
    let mut retained_indices = Vec::with_capacity(candidates.points.len());
    let report = retain_candidate_set_impl(candidates, filters, Some(&mut retained_indices))?;
    Ok((report, retained_indices))
}

fn retain_candidate_set_impl(
    candidates: &mut CandidateSet,
    filters: &[CandidateFilter],
    mut retained_indices: Option<&mut Vec<usize>>,
) -> Result<CandidateFilterReport, CandidateFilterError> {
    let input_count = candidates.points.len();
    if input_count == 0 || filters.is_empty() {
        if let Some(indices) = &mut retained_indices {
            indices.extend(0..input_count);
        }
        return Ok(CandidateFilterReport {
            input_count,
            retained_count: input_count,
        });
    }

    let columns = match candidate_column_indices(candidates) {
        Ok(columns) => columns,
        Err(CandidateSchemaError::EmptySet { .. }) => {
            unreachable!("a non-empty candidate set was validated as empty")
        }
        Err(error) => return Err(candidate_filter_schema_error(error)),
    };
    let offsets = filters
        .iter()
        .map(|filter| {
            columns.get(filter.column()).copied().ok_or_else(|| {
                CandidateFilterError::MissingColumn {
                    set: candidates.name.clone(),
                    column: filter.column().to_string(),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    for (point_index, point) in candidates.points.iter().enumerate() {
        for (filter, offset) in filters.iter().zip(&offsets) {
            if !point.values[*offset].1.is_finite() {
                return Err(CandidateFilterError::NonFiniteValue {
                    set: candidates.name.clone(),
                    point_index,
                    column: filter.column().to_string(),
                });
            }
        }
    }

    let mut original_index = 0;
    candidates.points.retain(|point| {
        let accepted = filters
            .iter()
            .zip(&offsets)
            .all(|(filter, offset)| filter.accepts_value(point.values[*offset].1));
        if accepted && let Some(indices) = &mut retained_indices {
            indices.push(original_index);
        }
        original_index += 1;
        accepted
    });
    Ok(CandidateFilterReport {
        input_count,
        retained_count: candidates.points.len(),
    })
}

fn candidate_filter_schema_error(error: CandidateSchemaError) -> CandidateFilterError {
    match error {
        CandidateSchemaError::EmptySet { .. } => {
            unreachable!("empty candidate sets are handled before schema validation")
        }
        CandidateSchemaError::DuplicateColumn {
            set,
            point_index,
            column,
        } => CandidateFilterError::DuplicateColumn {
            set,
            point_index,
            column,
        },
        CandidateSchemaError::ColumnCountMismatch {
            set,
            point_index,
            expected,
            actual,
        } => CandidateFilterError::ColumnCountMismatch {
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
        } => CandidateFilterError::ColumnNameMismatch {
            set,
            point_index,
            column_index,
            expected,
            actual,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CandidateFilter, CandidateFilterError, CandidateFilterReport, InvalidCandidateFilter,
        retain_candidate_set,
    };
    use crate::exploration::candidate::{
        CandidateEquality, CandidatePairJoin, CandidatePairSelection, CandidatePoint, CandidateSet,
    };

    fn candidate(width: f64, length: f64) -> CandidatePoint {
        CandidatePoint::new(vec![
            ("width".to_string(), width),
            ("length".to_string(), length),
        ])
    }

    fn candidates() -> CandidateSet {
        CandidateSet::new(
            "x1",
            vec![
                candidate(2.0, 1.0),
                candidate(5.0, 2.0),
                candidate(10.0, 3.0),
                candidate(12.0, 4.0),
            ],
        )
    }

    fn candidate_value_bits(candidates: &CandidateSet) -> Vec<Vec<(String, u64)>> {
        candidates
            .points
            .iter()
            .map(|point| {
                point
                    .values
                    .iter()
                    .map(|(name, value)| (name.clone(), value.to_bits()))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn at_most_is_inclusive_and_does_not_use_absolute_values() {
        let filter = CandidateFilter::at_most("width", 10.0).unwrap();

        assert!(filter.accepts_value(-20.0));
        assert!(filter.accepts_value(10.0));
        assert!(!filter.accepts_value(10.1));
        assert!(!filter.accepts_value(f64::NAN));
        assert!(!filter.accepts_value(f64::NEG_INFINITY));
        assert_eq!(filter.column(), "width");
        assert_eq!(filter.minimum(), None);
        assert_eq!(filter.maximum(), Some(10.0));
    }

    #[test]
    fn at_least_is_inclusive() {
        let filter = CandidateFilter::at_least("length", -2.0).unwrap();

        assert!(!filter.accepts_value(-2.1));
        assert!(filter.accepts_value(-2.0));
        assert!(filter.accepts_value(3.0));
    }

    #[test]
    fn between_accepts_both_boundaries() {
        let filter = CandidateFilter::between("voltage", -1.0, 1.0).unwrap();

        assert!(!filter.accepts_value(-1.1));
        assert!(filter.accepts_value(-1.0));
        assert!(filter.accepts_value(0.0));
        assert!(filter.accepts_value(1.0));
        assert!(!filter.accepts_value(1.1));
    }

    #[test]
    fn rejects_invalid_filter_definitions() {
        assert_eq!(
            CandidateFilter::at_most("", 1.0),
            Err(InvalidCandidateFilter::EmptyColumn)
        );
        assert_eq!(
            CandidateFilter::new("width", None, None),
            Err(InvalidCandidateFilter::MissingBounds {
                column: "width".to_string(),
            })
        );
        assert_eq!(
            CandidateFilter::at_least("width", f64::NAN),
            Err(InvalidCandidateFilter::NonFiniteMinimum {
                column: "width".to_string(),
            })
        );
        assert_eq!(
            CandidateFilter::at_most("width", f64::INFINITY),
            Err(InvalidCandidateFilter::NonFiniteMaximum {
                column: "width".to_string(),
            })
        );
        assert_eq!(
            CandidateFilter::between("width", 2.0, 1.0),
            Err(InvalidCandidateFilter::InvertedRange {
                column: "width".to_string(),
            })
        );
    }

    #[test]
    fn retains_candidates_in_place_with_an_inclusive_maximum() {
        let mut candidates = candidates();
        let report = retain_candidate_set(
            &mut candidates,
            &[CandidateFilter::at_most("width", 10.0).unwrap()],
        )
        .unwrap();

        assert_eq!(
            report,
            CandidateFilterReport {
                input_count: 4,
                retained_count: 3,
            }
        );
        assert_eq!(report.rejected_count(), 1);
        assert_eq!(
            candidates
                .points
                .iter()
                .map(|point| point.get("width").unwrap())
                .collect::<Vec<_>>(),
            vec![2.0, 5.0, 10.0]
        );
    }

    #[test]
    fn combines_multiple_filters_with_logical_and() {
        let mut candidates = candidates();
        let report = retain_candidate_set(
            &mut candidates,
            &[
                CandidateFilter::at_least("width", 5.0).unwrap(),
                CandidateFilter::between("length", 2.0, 3.0).unwrap(),
            ],
        )
        .unwrap();

        assert_eq!(report.retained_count(), 2);
        assert_eq!(
            candidates.points,
            vec![candidate(5.0, 2.0), candidate(10.0, 3.0)]
        );
    }

    #[test]
    fn can_reject_every_candidate() {
        let mut candidates = candidates();
        let report = retain_candidate_set(
            &mut candidates,
            &[CandidateFilter::at_most("width", 1.0).unwrap()],
        )
        .unwrap();

        assert_eq!(report.input_count(), 4);
        assert_eq!(report.retained_count(), 0);
        assert_eq!(report.rejected_count(), 4);
        assert!(candidates.points.is_empty());
    }

    #[test]
    fn empty_inputs_are_no_op_operations() {
        let mut candidates = candidates();
        let unchanged = retain_candidate_set(&mut candidates, &[]).unwrap();
        assert_eq!(unchanged.input_count(), 4);
        assert_eq!(unchanged.retained_count(), 4);

        let mut empty = CandidateSet::new("empty", Vec::new());
        let empty_report = retain_candidate_set(
            &mut empty,
            &[CandidateFilter::at_most("missing", 1.0).unwrap()],
        )
        .unwrap();
        assert_eq!(empty_report, CandidateFilterReport::default());
    }

    #[test]
    fn missing_columns_leave_the_candidate_set_unchanged() {
        let mut candidates = candidates();
        let original = candidates.clone();

        assert_eq!(
            retain_candidate_set(
                &mut candidates,
                &[CandidateFilter::at_most("missing", 10.0).unwrap()],
            ),
            Err(CandidateFilterError::MissingColumn {
                set: "x1".to_string(),
                column: "missing".to_string(),
            })
        );
        assert_eq!(candidates, original);
    }

    #[test]
    fn non_finite_values_leave_the_candidate_set_unchanged() {
        let mut candidates = candidates();
        candidates.points[2].values[0].1 = f64::NAN;
        let original = candidate_value_bits(&candidates);

        assert_eq!(
            retain_candidate_set(
                &mut candidates,
                &[CandidateFilter::at_most("width", 10.0).unwrap()],
            ),
            Err(CandidateFilterError::NonFiniteValue {
                set: "x1".to_string(),
                point_index: 2,
                column: "width".to_string(),
            })
        );
        assert_eq!(candidate_value_bits(&candidates), original);
    }

    #[test]
    fn inconsistent_schemas_leave_the_candidate_set_unchanged() {
        let mut candidates = candidates();
        candidates.points[2].values[1].0 = "other".to_string();
        let original = candidates.clone();

        assert_eq!(
            retain_candidate_set(
                &mut candidates,
                &[CandidateFilter::at_most("width", 10.0).unwrap()],
            ),
            Err(CandidateFilterError::ColumnNameMismatch {
                set: "x1".to_string(),
                point_index: 2,
                column_index: 1,
                expected: "length".to_string(),
                actual: "other".to_string(),
            })
        );
        assert_eq!(candidates, original);
    }

    #[test]
    fn filtered_candidate_indices_are_used_by_the_indexed_join() {
        let point = |voltage_column: &str, voltage: f64, width: f64| {
            CandidatePoint::new(vec![
                (voltage_column.to_string(), voltage),
                ("width".to_string(), width),
            ])
        };
        let mut left = CandidateSet::new(
            "xdp",
            vec![
                point("xdp.voutp", 0.8, 12.0),
                point("xdp.voutp", 0.8, 5.0),
                point("xdp.voutp", 0.9, 6.0),
                point("xdp.voutp", 1.0, 15.0),
            ],
        );
        let mut right = CandidateSet::new(
            "xcm",
            vec![
                point("xcm.voutp", 0.7, 4.0),
                point("xcm.voutp", 0.8, 11.0),
                point("xcm.voutp", 0.8, 8.0),
                point("xcm.voutp", 0.9, 9.0),
            ],
        );

        let maximum_width = CandidateFilter::at_most("width", 10.0).unwrap();
        assert_eq!(
            retain_candidate_set(&mut left, std::slice::from_ref(&maximum_width))
                .unwrap()
                .retained_count(),
            2
        );
        assert_eq!(
            retain_candidate_set(&mut right, &[maximum_width])
                .unwrap()
                .retained_count(),
            3
        );

        let selections = CandidatePairJoin::new(
            &left,
            &right,
            &[CandidateEquality::new("xdp.voutp", "xcm.voutp")],
        )
        .unwrap()
        .collect::<Vec<_>>();

        assert_eq!(
            selections,
            vec![
                CandidatePairSelection::new(0, 1),
                CandidatePairSelection::new(1, 2),
            ]
        );
        for selection in selections {
            assert_eq!(
                left.points[selection.left_index].get("xdp.voutp"),
                right.points[selection.right_index].get("xcm.voutp")
            );
            assert!(left.points[selection.left_index].get("width").unwrap() <= 10.0);
            assert!(right.points[selection.right_index].get("width").unwrap() <= 10.0);
        }
    }
}
