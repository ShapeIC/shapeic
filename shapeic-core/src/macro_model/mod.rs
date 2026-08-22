//! Reusable macros, compact circuit models, and their exploration definitions.

mod catalog;
mod model;

pub use catalog::{MacroCatalog, MacroCatalogError};
pub use model::{
    Macro, MacroAcTestbench, MacroExploration, MacroPort, MacroPortRole, MacroTestbenchSource,
};
