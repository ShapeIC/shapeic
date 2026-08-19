use std::error::Error;
use std::fmt;

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

#[cfg(test)]
mod tests {
    use super::{CandidateFilter, InvalidCandidateFilter};

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
}
