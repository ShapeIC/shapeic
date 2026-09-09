#[derive(Debug, Clone, PartialEq)]
pub struct ExplorationColumn {
    pub name: String,
    pub values: Vec<f64>,
}

impl ExplorationColumn {
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct ExplorationTable {
    pub columns: Vec<ExplorationColumn>,
    pub row_count: usize,
}

impl ExplorationTable {
    pub fn column(&self, name: &str) -> Option<&ExplorationColumn> {
        self.columns.iter().find(|column| column.name == name)
    }

    pub fn add_sum_column(
        &mut self,
        name: impl Into<String>,
        source_columns: &[&str],
    ) -> Result<(), ExplorationTableError> {
        let name = name.into();
        let mut values = vec![0.0; self.row_count];

        for source_column in source_columns {
            let column =
                self.column(source_column)
                    .ok_or_else(|| ExplorationTableError::MissingColumn {
                        column: (*source_column).to_string(),
                    })?;

            if column.values.len() != self.row_count {
                return Err(ExplorationTableError::ColumnLengthMismatch {
                    column: column.name.clone(),
                    expected: self.row_count,
                    actual: column.values.len(),
                });
            }

            for (total, value) in values.iter_mut().zip(&column.values) {
                *total += value;
            }
        }

        self.columns.push(ExplorationColumn { name, values });
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplorationTableError {
    MissingColumn {
        column: String,
    },
    MaskLengthMismatch {
        column: String,
        mask_len: usize,
        values_len: usize,
    },
    ColumnLengthMismatch {
        column: String,
        expected: usize,
        actual: usize,
    },
}
