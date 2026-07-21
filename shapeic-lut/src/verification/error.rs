use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Errors that prevent a verification batch from being configured or stored.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VerificationError {
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: io::Error,
    },

    #[error("failed to write verification CSV: {0}")]
    Csv(#[from] csv::Error),

    #[error("invalid verification configuration: {0}")]
    InvalidConfig(String),

    #[error("invalid verification template: {0}")]
    Template(String),

    #[error("verification output directory '{}' already exists", .0.display())]
    OutputExists(PathBuf),
}

impl VerificationError {
    pub(crate) fn io(context: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}
