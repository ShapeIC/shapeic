//! Native Rust access to SSTADEx four-dimensional and Shapeic five-dimensional MOS LUTs.
//!
//! The crate preserves the source arrays as [`DType::F32`] or [`DType::F64`],
//! exposes their metadata, and evaluates [`Expr`] trees with four- or
//! five-dimensional linear interpolation.
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
//!
//! Width-dependent Shapeic LUTs are queried explicitly with [`LutPoint`]:
//!
//! ```no_run
//! use shapeic_lut::{LookupTable, LutPoint, OperatingPoint};
//!
//! let table = LookupTable::open("nmos-5d.npz")?;
//! let model = table.model("sg13_lv_nmos")?;
//! let point = LutPoint::new(
//!     OperatingPoint::new(0.4e-6, 0.0, 0.6, 0.6),
//!     0.5e-6,
//! );
//! let id = model.query_parameter_at(&point, "id")?;
//! # let _ = id;
//! # Ok::<(), shapeic_lut::LutError>(())
//! ```
//!
//! [`DeviceLut::size_for_current`] selects a per-finger width and integer finger count for a
//! requested total current. Any expressions passed to it are returned as per-finger values.

#![warn(missing_docs)]

mod array;
mod error;
mod expression;
mod interpolation;
mod model;
mod npz;
mod sizing;
pub mod verification;

pub use array::{DType, LutArray};
pub use error::LutError;
pub use expression::{Expr, MosExpression};
pub use model::{
    Axis, DeviceLut, LookupTable, LutMetadata, LutPoint, MosCapacitanceMatrix,
    MosExtrinsicCapacitances, OperatingPoint,
};
pub use sizing::CurrentSizingResult;
