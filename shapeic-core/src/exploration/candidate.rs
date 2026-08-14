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

    Ok(CandidateSet::new(name, points))
}

pub fn candidate_column_name(prefix: &str, column: &str) -> String {
    format!("{prefix}.{column}")
}
