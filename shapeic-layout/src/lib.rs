//! Physical primitive lookup tables for layout-aware Shapeic analysis.

mod archive;
mod error;
mod interpolation;
mod model;
mod validation;

pub use error::LayoutError;
pub use model::{
    PhysicalLookupTable, PhysicalMetadata, PhysicalPoint, PhysicalPrimitive, PortAdmittance,
};
