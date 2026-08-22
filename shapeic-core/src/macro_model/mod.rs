//! Reusable macros, compact circuit models, and their exploration definitions.

mod catalog;
mod evaluate;
mod model;
mod prepare;
mod render;
mod validation;

pub use catalog::{MacroCatalog, MacroCatalogError};
pub use evaluate::{
    MacroAcCandidateAnalysisError, PreparedMacroAcCandidateEvaluator,
    PreparedMacroAcCandidateEvaluatorError,
};
pub use model::{
    Macro, MacroAcTestbench, MacroExploration, MacroPort, MacroPortRole, MacroTestbenchSource,
};
pub use prepare::{
    MacroTestbenchPrepareError, PreparedMacroAcTestbench, prepare_macro_ac_testbench,
};
pub use render::{
    ExpandedSmallSignalNetlist, MacroRenderError, MacroRenderMode, ResolvedPrimitiveBranch,
    render_expanded_small_signal_netlist, render_small_signal_netlist,
};
pub use validation::{
    MacroCircuitKind, MacroValidationError, validate_macro, validate_macro_catalog,
};
