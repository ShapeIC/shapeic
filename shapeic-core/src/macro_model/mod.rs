//! Reusable macros, compact circuit models, and their exploration definitions.

mod catalog;
mod model;
mod validation;

pub use catalog::{MacroCatalog, MacroCatalogError};
pub use model::{
    Macro, MacroAcTestbench, MacroExploration, MacroPort, MacroPortRole, MacroTestbenchSource,
};
pub use validation::{
    MacroCircuitKind, MacroValidationError, validate_macro, validate_macro_catalog,
};
