use std::collections::HashMap;

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
        CandidatePairSelection, CandidatePoint, CandidateSchemaError, CandidateSet,
        candidate_column_indices,
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
