use std::io;

use thiserror::Error;

use crate::Axis;

/// Errors produced while loading or querying a lookup table.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LutError {
    #[error("failed to access LUT data: {0}")]
    Io(#[from] io::Error),

    #[error("invalid NPZ archive: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("invalid lookup_table.npy header: {0}")]
    NpyHeader(#[from] ndarray_npy::npy::header::ReadHeaderError),

    #[error("invalid lookup table pickle: {0}")]
    Pickle(#[from] serde_pickle::Error),

    #[error("invalid SSTADEx LUT schema at {context}: {reason}")]
    Schema { context: String, reason: String },

    #[error("unsupported LUT data at {context}: {reason}")]
    Unsupported { context: String, reason: String },

    #[error("model '{0}' was not found in the lookup table")]
    UnknownModel(String),

    #[error("parameter '{parameter}' was not found in model '{model}'")]
    UnknownParameter { model: String, parameter: String },

    #[error("device parameter '{parameter}' was not found in model '{model}'")]
    UnknownDeviceParameter { model: String, parameter: String },

    #[error(
        "{axis} value {value} is outside the LUT range [{minimum}, {maximum}] for model '{model}'"
    )]
    OutOfRange {
        model: String,
        axis: Axis,
        value: f64,
        minimum: f64,
        maximum: f64,
    },

    #[error("non-finite {context} in model '{model}'")]
    NonFinite { model: String, context: String },

    #[error("batch query failed at point {point_index}, expression {expression_index}: {source}")]
    Batch {
        point_index: usize,
        expression_index: usize,
        #[source]
        source: Box<LutError>,
    },
}

impl LutError {
    pub(crate) fn schema(context: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Schema {
            context: context.into(),
            reason: reason.into(),
        }
    }

    pub(crate) fn unsupported(context: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Unsupported {
            context: context.into(),
            reason: reason.into(),
        }
    }
}
