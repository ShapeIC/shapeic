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

    #[error("invalid NPY header: {0}")]
    NpyHeader(#[from] ndarray_npy::npy::header::ReadHeaderError),

    #[error("invalid lookup table pickle: {0}")]
    Pickle(#[from] serde_pickle::Error),

    #[error("invalid Shapeic LUT manifest: {0}")]
    Manifest(#[from] serde_json::Error),

    #[error("invalid LUT schema at {context}: {reason}")]
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

    #[error("model '{model}' is five-dimensional; query it with a LutPoint")]
    FingerWidthRequired { model: String },

    #[error("model '{model}' has no finger_width axis")]
    NoFingerWidthAxis { model: String },

    #[error(
        "finger_width value {value} is outside the LUT range [{minimum}, {maximum}] for model '{model}'"
    )]
    FingerWidthOutOfRange {
        model: String,
        value: f64,
        minimum: f64,
        maximum: f64,
    },

    #[error("non-finite {context} in model '{model}'")]
    NonFinite { model: String, context: String },

    #[error(
        "requested drain current must be finite and positive for model '{model}', found {value}"
    )]
    InvalidCurrentTarget { model: String, value: f64 },

    #[error(
        "drain current must be positive for inverse sizing of model '{model}', found {current} at finger_width={finger_width}"
    )]
    NonPositiveFingerCurrent {
        model: String,
        finger_width: f64,
        current: f64,
    },

    #[error(
        "drain current is not strictly increasing with finger width for model '{model}': Id({lower_width})={lower_current}, Id({upper_width})={upper_current}"
    )]
    NonMonotonicFingerCurrent {
        model: String,
        lower_width: f64,
        lower_current: f64,
        upper_width: f64,
        upper_current: f64,
    },

    #[error(
        "requested drain current {requested_current} requires more than {maximum_fingers} fingers for model '{model}'"
    )]
    FingerCountOverflow {
        model: String,
        requested_current: f64,
        maximum_fingers: u32,
    },

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
