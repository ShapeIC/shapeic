//! Reusable macros, compact circuit models, and their exploration definitions.

mod catalog;
mod candidates;
mod combination;
mod evaluate;
mod input;
mod model;
mod prepare;
mod render;
mod validation;

pub use catalog::{MacroCatalog, MacroCatalogError};
pub use candidates::{
    MacroCandidateBuildError, MacroCandidateSets, MacroInstanceCandidateSet,
    build_macro_candidate_sets,
};
pub use combination::{
    MacroCandidateCombinationError, MacroCandidateCombinationJoin,
    MacroCandidateCombinationPlan, plan_macro_candidate_combinations,
};
pub use evaluate::{
    MacroAcCandidateAnalysisError, PreparedMacroAcCandidateEvaluator,
    PreparedMacroAcCandidateEvaluatorError,
};
pub use input::{
    CompactMacroInstanceExplorationInput, MacroExplorationInput,
    MacroExplorationInputRegistrationError, MacroExplorationInputValidationError,
    MacroExplorationInstanceKind, PrimitiveInstanceExplorationInput,
    validate_macro_exploration_input,
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
