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
pub struct CandidateEquality {
    pub left_column: String,
    pub right_column: String,
}

impl CandidateEquality {
    pub fn new(left_column: impl Into<String>, right_column: impl Into<String>) -> Self {
        Self {
            left_column: left_column.into(),
            right_column: right_column.into(),
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
    MissingColumn {
        set: String,
        column: String,
    },
    NonFiniteJoinValue {
        set: String,
        point_index: usize,
        column: String,
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
    left: &'a CandidateSet,
    right: &'a CandidateSet,
    state: CandidatePairJoinState,
    remaining: usize,
}

#[derive(Debug)]
enum CandidatePairJoinState {
    Cartesian {
        next_left: usize,
        next_right: usize,
    },
    Indexed {
        indexed_side: IndexedSide,
        index: HashMap<Box<[u64]>, Vec<usize>>,
        probe_offsets: Vec<usize>,
        next_probe: usize,
        current_probe: usize,
        current_key: Option<Box<[u64]>>,
        next_match: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexedSide {
    Left,
    Right,
}

impl<'a> CandidatePairJoin<'a> {
    pub fn new(
        left: &'a CandidateSet,
        right: &'a CandidateSet,
        equalities: &[CandidateEquality],
    ) -> Result<Self, CandidateJoinError> {
        let left_columns = candidate_column_indices(left)?;
        let right_columns = candidate_column_indices(right)?;

        if equalities.is_empty() {
            let remaining = checked_pair_count(left, right)?;
            return Ok(Self {
                left,
                right,
                state: CandidatePairJoinState::Cartesian {
                    next_left: 0,
                    next_right: 0,
                },
                remaining,
            });
        }

        let left_offsets = equality_offsets(left, &left_columns, equalities, true)?;
        let right_offsets = equality_offsets(right, &right_columns, equalities, false)?;
        let left_names = equality_names(equalities, true);
        let right_names = equality_names(equalities, false);
        let indexed_side = if left.points.len() < right.points.len() {
            IndexedSide::Left
        } else {
            IndexedSide::Right
        };
        let (indexed_set, indexed_offsets, indexed_names, probe_set, probe_offsets, probe_names) =
            match indexed_side {
                IndexedSide::Left => (
                    left,
                    &left_offsets,
                    &left_names,
                    right,
                    right_offsets,
                    right_names,
                ),
                IndexedSide::Right => (
                    right,
                    &right_offsets,
                    &right_names,
                    left,
                    left_offsets,
                    left_names,
                ),
            };

        let mut index = HashMap::<Box<[u64]>, Vec<usize>>::new();
        for (point_index, point) in indexed_set.points.iter().enumerate() {
            let key = candidate_join_key(
                indexed_set,
                point,
                point_index,
                indexed_offsets,
                indexed_names,
            )?;
            index.entry(key).or_default().push(point_index);
        }

        let mut remaining = 0usize;
        for (point_index, point) in probe_set.points.iter().enumerate() {
            let key =
                candidate_join_key(probe_set, point, point_index, &probe_offsets, &probe_names)?;
            if let Some(matches) = index.get(&key) {
                remaining = remaining.checked_add(matches.len()).ok_or(
                    CandidateJoinError::PairCountOverflow {
                        left_count: left.points.len(),
                        right_count: right.points.len(),
                    },
                )?;
            }
        }

        Ok(Self {
            left,
            right,
            state: CandidatePairJoinState::Indexed {
                indexed_side,
                index,
                probe_offsets,
                next_probe: 0,
                current_probe: 0,
                current_key: None,
                next_match: 0,
            },
            remaining,
        })
    }

    pub fn cartesian(
        left: &'a CandidateSet,
        right: &'a CandidateSet,
    ) -> Result<Self, CandidateJoinError> {
        Self::new(left, right, &[])
    }
}

impl Iterator for CandidatePairJoin<'_> {
    type Item = CandidatePairSelection;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let selection = match &mut self.state {
            CandidatePairJoinState::Cartesian {
                next_left,
                next_right,
            } => {
                let selection = CandidatePairSelection::new(*next_left, *next_right);
                *next_right += 1;
                if *next_right == self.right.points.len() {
                    *next_right = 0;
                    *next_left += 1;
                }
                selection
            }
            CandidatePairJoinState::Indexed {
                indexed_side,
                index,
                probe_offsets,
                next_probe,
                current_probe,
                current_key,
                next_match,
            } => loop {
                if let Some(key) = current_key.as_ref() {
                    let matches = index
                        .get(key)
                        .expect("the current indexed join key must have a bucket");
                    if let Some(indexed_index) = matches.get(*next_match).copied() {
                        *next_match += 1;
                        break match indexed_side {
                            IndexedSide::Left => {
                                CandidatePairSelection::new(indexed_index, *current_probe)
                            }
                            IndexedSide::Right => {
                                CandidatePairSelection::new(*current_probe, indexed_index)
                            }
                        };
                    }
                    *current_key = None;
                }

                let probe_set = match indexed_side {
                    IndexedSide::Left => self.right,
                    IndexedSide::Right => self.left,
                };
                let point = probe_set.points.get(*next_probe).expect(
                    "remaining indexed join pairs require another validated probe candidate",
                );
                *current_probe = *next_probe;
                *next_probe += 1;
                *next_match = 0;
                let key = candidate_join_key_unchecked(point, probe_offsets);
                if index.contains_key(&key) {
                    *current_key = Some(key);
                }
            },
        };
        self.remaining -= 1;
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

fn checked_pair_count(
    left: &CandidateSet,
    right: &CandidateSet,
) -> Result<usize, CandidateJoinError> {
    left.points
        .len()
        .checked_mul(right.points.len())
        .ok_or(CandidateJoinError::PairCountOverflow {
            left_count: left.points.len(),
            right_count: right.points.len(),
        })
}

fn equality_offsets(
    candidates: &CandidateSet,
    columns: &HashMap<String, usize>,
    equalities: &[CandidateEquality],
    left: bool,
) -> Result<Vec<usize>, CandidateJoinError> {
    equalities
        .iter()
        .map(|equality| {
            let column = if left {
                &equality.left_column
            } else {
                &equality.right_column
            };
            columns
                .get(column)
                .copied()
                .ok_or_else(|| CandidateJoinError::MissingColumn {
                    set: candidates.name.clone(),
                    column: column.clone(),
                })
        })
        .collect()
}

fn equality_names(equalities: &[CandidateEquality], left: bool) -> Vec<String> {
    equalities
        .iter()
        .map(|equality| {
            if left {
                equality.left_column.clone()
            } else {
                equality.right_column.clone()
            }
        })
        .collect()
}

fn candidate_join_key(
    candidates: &CandidateSet,
    point: &CandidatePoint,
    point_index: usize,
    offsets: &[usize],
    columns: &[String],
) -> Result<Box<[u64]>, CandidateJoinError> {
    offsets
        .iter()
        .zip(columns)
        .map(|(offset, column)| {
            let value = point.values[*offset].1;
            canonical_join_value(value).ok_or_else(|| CandidateJoinError::NonFiniteJoinValue {
                set: candidates.name.clone(),
                point_index,
                column: column.clone(),
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

fn candidate_join_key_unchecked(point: &CandidatePoint, offsets: &[usize]) -> Box<[u64]> {
    offsets
        .iter()
        .map(|offset| {
            canonical_join_value(point.values[*offset].1)
                .expect("indexed join values were validated during construction")
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn canonical_join_value(value: f64) -> Option<u64> {
    value
        .is_finite()
        .then(|| if value == 0.0 { 0 } else { value.to_bits() })
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
        CandidateEquality, CandidateJoinError, CandidatePairJoin, CandidatePairSelection,
        CandidatePoint, CandidateSchemaError, CandidateSet, candidate_column_indices,
    };

    fn point(values: &[(&str, f64)]) -> CandidatePoint {
        CandidatePoint::new(
            values
                .iter()
                .map(|(name, value)| ((*name).to_owned(), *value))
                .collect(),
        )
    }

    fn voltage_length_candidates(
        name: &str,
        voltage_column: &str,
        length_column: &str,
        voltages: &[f64],
        lengths: &[f64],
    ) -> CandidateSet {
        let points = voltages
            .iter()
            .flat_map(|voltage| {
                lengths.iter().map(|length| {
                    CandidatePoint::new(vec![
                        (voltage_column.to_owned(), *voltage),
                        (length_column.to_owned(), *length),
                    ])
                })
            })
            .collect();
        CandidateSet::new(name, points)
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
    fn indexed_join_reduces_four_voltages_and_three_lengths_to_compatible_pairs() {
        let voltages = [0.8, 0.9, 1.0, 1.1];
        let lengths = [0.4e-6, 0.8e-6, 1.6e-6];
        let left =
            voltage_length_candidates("xdp", "xdp.voutp", "length__xdp__m1", &voltages, &lengths);
        let right =
            voltage_length_candidates("xcm", "xcm.voutp", "length__xcm__m1", &voltages, &lengths);
        let equality = CandidateEquality::new("xdp.voutp", "xcm.voutp");
        let mut join = CandidatePairJoin::new(&left, &right, &[equality]).unwrap();

        assert_eq!(left.points.len() * right.points.len(), 144);
        assert_eq!(join.len(), 36);
        let selections = join.by_ref().collect::<Vec<_>>();
        assert_eq!(join.next(), None);
        for selection in &selections {
            assert_eq!(
                left.points[selection.left_index].get("xdp.voutp"),
                right.points[selection.right_index].get("xcm.voutp")
            );
        }
        for voltage in voltages {
            assert_eq!(
                selections
                    .iter()
                    .filter(|selection| {
                        left.points[selection.left_index].get("xdp.voutp") == Some(voltage)
                    })
                    .count(),
                9
            );
        }
    }

    #[test]
    fn indexed_join_uses_a_composite_key_for_multiple_shared_nodes() {
        let left = CandidateSet::new(
            "xdp",
            vec![
                point(&[("xdp.voutp", 0.8), ("xdp.voutn", 0.4)]),
                point(&[("xdp.voutp", 0.8), ("xdp.voutn", 0.5)]),
                point(&[("xdp.voutp", 0.9), ("xdp.voutn", 0.4)]),
            ],
        );
        let right = CandidateSet::new(
            "xcm",
            vec![
                point(&[("xcm.voutp", 0.8), ("xcm.vinp", 0.4)]),
                point(&[("xcm.voutp", 0.8), ("xcm.vinp", 0.6)]),
                point(&[("xcm.voutp", 0.9), ("xcm.vinp", 0.4)]),
            ],
        );
        let equalities = [
            CandidateEquality::new("xdp.voutp", "xcm.voutp"),
            CandidateEquality::new("xdp.voutn", "xcm.vinp"),
        ];

        assert_eq!(
            CandidatePairJoin::new(&left, &right, &equalities)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![
                CandidatePairSelection::new(0, 0),
                CandidatePairSelection::new(2, 2),
            ]
        );
    }

    #[test]
    fn indexed_join_preserves_selection_orientation_when_the_left_set_is_indexed() {
        let left = CandidateSet::new("xdp", vec![point(&[("xdp.voutp", 1.0)])]);
        let right = CandidateSet::new(
            "xcm",
            vec![point(&[("xcm.voutp", 1.0)]), point(&[("xcm.voutp", 1.0)])],
        );

        assert_eq!(
            CandidatePairJoin::new(
                &left,
                &right,
                &[CandidateEquality::new("xdp.voutp", "xcm.voutp")],
            )
            .unwrap()
            .collect::<Vec<_>>(),
            vec![
                CandidatePairSelection::new(0, 0),
                CandidatePairSelection::new(0, 1),
            ]
        );
    }

    #[test]
    fn indexed_join_canonicalizes_signed_zero_and_can_produce_no_matches() {
        let left = CandidateSet::new(
            "xdp",
            vec![point(&[("xdp.voutp", -0.0)]), point(&[("xdp.voutp", 1.0)])],
        );
        let right = CandidateSet::new(
            "xcm",
            vec![point(&[("xcm.voutp", 0.0)]), point(&[("xcm.voutp", 2.0)])],
        );
        let equality = CandidateEquality::new("xdp.voutp", "xcm.voutp");

        assert_eq!(
            CandidatePairJoin::new(&left, &right, &[equality])
                .unwrap()
                .collect::<Vec<_>>(),
            vec![CandidatePairSelection::new(0, 0)]
        );

        let unmatched_left = CandidateSet::new("a", vec![point(&[("a.v", 1.0)])]);
        let unmatched_right = CandidateSet::new("b", vec![point(&[("b.v", 2.0)])]);
        let no_matches = CandidatePairJoin::new(
            &unmatched_left,
            &unmatched_right,
            &[CandidateEquality::new("a.v", "b.v")],
        )
        .unwrap();
        assert_eq!(no_matches.len(), 0);
    }

    #[test]
    fn indexed_join_rejects_missing_or_non_finite_shared_node_columns() {
        let left = CandidateSet::new("xdp", vec![point(&[("xdp.voutp", f64::NAN)])]);
        let right = CandidateSet::new("xcm", vec![point(&[("xcm.voutp", 1.0)])]);

        assert_eq!(
            CandidatePairJoin::new(
                &left,
                &right,
                &[CandidateEquality::new("xdp.missing", "xcm.voutp")],
            )
            .unwrap_err(),
            CandidateJoinError::MissingColumn {
                set: "xdp".to_owned(),
                column: "xdp.missing".to_owned(),
            }
        );
        assert_eq!(
            CandidatePairJoin::new(
                &left,
                &right,
                &[CandidateEquality::new("xdp.voutp", "xcm.voutp")],
            )
            .unwrap_err(),
            CandidateJoinError::NonFiniteJoinValue {
                set: "xdp".to_owned(),
                point_index: 0,
                column: "xdp.voutp".to_owned(),
            }
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
