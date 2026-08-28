//! Reusable macros, compact circuit models, and their exploration definitions.

mod catalog;
mod candidates;
mod combination;
mod execution;
mod evaluate;
mod explore;
mod input;
mod model;
mod physical;
mod prepare;
mod projection;
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
pub use execution::{
    ElectricalAnalysisExecution, MacroExecutionConfig, MacroExecutionConfigError,
};
pub use evaluate::{
    MacroAcCandidateAnalysisError, MacroAcCandidateEvaluation, PreparedMacroAcCandidateEvaluator,
    PreparedMacroAcCandidateEvaluatorError,
};
pub use explore::{
    MacroAcExplorationError, MacroAcRejectionCounts, MacroAcTestbenchOutcome,
    MacroAcTestbenchStatistics, MacroAcceptedCandidate, MacroExplorationError,
    MacroExplorationResult, MacroExplorationStatistics, explore_macro_ac_candidates,
    explore_macro_ac_candidates_with_physical_lut,
};
pub use input::{
    CompactMacroInstanceExplorationInput, MacroExplorationInput,
    MacroExplorationInputRegistrationError, MacroExplorationInputValidationError,
    MacroExplorationInstanceKind, PrimitiveInstanceExplorationInput,
    validate_macro_exploration_input,
};
pub use model::{
    Macro, MacroAcTestbench, MacroAnalysisDomain, MacroCompactOutputBinding, MacroExploration,
    MacroInterfaceBinding, MacroOutputSource, MacroPort, MacroPortRole, MacroTestbenchSource,
};
pub use physical::CandidatePhysicalBindingError;
pub(crate) use physical::{CandidatePhysicalBinder, PhysicalCandidateStampOutcome};
pub use prepare::{
    MacroTestbenchPrepareError, PreparedMacroAcTestbench, prepare_macro_ac_testbench,
};
pub use projection::{
    MacroCandidateProjection, MacroCandidateProjectionError, MacroOutputResolutionError,
};
pub use render::{
    ExpandedSmallSignalNetlist, MacroRenderError, MacroRenderMode,
    ResolvedPhysicalCandidateColumns, ResolvedPhysicalPort, ResolvedPhysicalPrimitive,
    ResolvedPrimitiveBranch, render_expanded_small_signal_netlist, render_small_signal_netlist,
};
pub use validation::{
    MacroCircuitKind, MacroValidationError, validate_macro, validate_macro_catalog,
};
