//! Native Rust access to SSTADEx MOS lookup tables.
//!
//! The crate preserves the source arrays as [`DType::F32`] or [`DType::F64`],
//! exposes their metadata, and evaluates [`Expr`] trees with four-dimensional
//! linear interpolation over [`OperatingPoint`] coordinates.
//!
//! ```no_run
//! use shapeic_lut::{LookupTable, MosExpression, OperatingPoint};
//!
//! let table = LookupTable::open("nmos.npz")?;
//! let model = table.model("sg13_lv_nmos")?;
//! let point = OperatingPoint::new(0.4e-6, 0.0, 0.6, 0.6);
//! let gmid = model.standard_expression(MosExpression::GmOverId)?;
//! let value = model.query_expression(&point, &gmid)?;
//! # let _ = value;
//! # Ok::<(), shapeic_lut::LutError>(())
//! ```

mod array;
mod error;
mod expression;
mod interpolation;
mod model;
mod npz;
pub mod verification;

pub use array::{DType, LutArray};
pub use error::LutError;
pub use expression::{Expr, MosExpression};
pub use model::{Axis, DeviceLut, LookupTable, LutMetadata, OperatingPoint};
