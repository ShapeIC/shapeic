use thiserror::Error;

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("NPY error: {0}")]
    Npy(#[from] ndarray_npy::ReadNpyError),
    #[error("physical LUT schema error in {context}: {message}")]
    Schema { context: String, message: String },
    #[error("unknown physical primitive '{0}'")]
    UnknownPrimitive(String),
    #[error(
        "physical point is outside primitive '{primitive}' axis '{axis}': {value} not in [{minimum}, {maximum}]"
    )]
    OutOfPhysicalRange {
        primitive: String,
        axis: &'static str,
        value: f64,
        minimum: f64,
        maximum: f64,
    },
    #[error("nf must be greater than zero")]
    InvalidFingerCount,
}

impl LayoutError {
    pub(crate) fn schema(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Schema {
            context: context.into(),
            message: message.into(),
        }
    }
}
