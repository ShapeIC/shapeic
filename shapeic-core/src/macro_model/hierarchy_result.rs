//! Immutable hierarchical results, derivation audit, and retention policy.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use super::{
    MacroDerivationPruneReason, MacroExecutionReport, MacroExplorationResult,
    MacroExplorationStatistics, MacroHierarchyPath, ResolvedChildConditions,
};

/// Controls whether complete provisional parent results remain available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MacroHierarchyRetentionPolicy {
    /// Retains preview statistics but releases provisional candidate sets.
    #[default]
    Summaries,
    /// Retains complete preview results for debugging and comparison.
    FullPreviews,
}

/// Final execution state of one concrete macro occurrence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroHierarchyNodeStatus {
    /// This route was never executed because an ancestor blocked traversal.
    NotReached {
        /// Closest rejected or pruned ancestor known to block this route.
        blocked_by: Option<MacroHierarchyPath>,
    },
    /// A macro without submacros was evaluated once.
    ExploredLeaf,
    /// Traversal stopped at a compact child without exploring its implementation.
    BlackBox {
        /// Number of aligned compact seed candidates exposed to the parent.
        candidates: usize,
    },
    /// A parent preview accepted no rows, so its children were not explored.
    PreviewRejected,
    /// Parent-derived conditions for this child have an empty intersection.
    DerivationPruned {
        /// Exact contradiction that pruned this occurrence.
        reason: MacroDerivationPruneReason,
    },
    /// A parent was evaluated definitively after projecting explored children.
    Refreshed,
    /// The first evaluation was definitive because every reached child was a blackbox.
    PreviewFinalized,
}

/// Summary of one provisional parent evaluation.
#[derive(Debug)]
pub struct MacroHierarchyPreviewRecord {
    compatible_candidates: usize,
    accepted_candidates: usize,
    statistics: MacroExplorationStatistics,
    execution: MacroExecutionReport,
    retained_result: Option<Arc<MacroExplorationResult>>,
}

impl MacroHierarchyPreviewRecord {
    pub(super) fn new(
        result: &Arc<MacroExplorationResult>,
        retention: MacroHierarchyRetentionPolicy,
    ) -> Self {
        Self {
            compatible_candidates: result.statistics().compatible_candidates(),
            accepted_candidates: result.accepted().len(),
            statistics: result.statistics().clone(),
            execution: result.execution_report().clone(),
            retained_result: (retention == MacroHierarchyRetentionPolicy::FullPreviews)
                .then(|| Arc::clone(result)),
        }
    }

    /// Returns the number of connectivity-compatible preview candidates.
    pub const fn compatible_candidates(&self) -> usize {
        self.compatible_candidates
    }

    /// Returns the number of candidates accepted by the preview.
    pub const fn accepted_candidates(&self) -> usize {
        self.accepted_candidates
    }

    /// Returns the preview's analysis and rejection statistics.
    pub const fn statistics(&self) -> &MacroExplorationStatistics {
        &self.statistics
    }

    /// Returns the preview's execution policy, timing, and workload report.
    pub const fn execution_report(&self) -> &MacroExecutionReport {
        &self.execution
    }

    /// Returns the complete provisional result when full retention is enabled.
    pub fn retained_result(&self) -> Option<&Arc<MacroExplorationResult>> {
        self.retained_result.as_ref()
    }
}

/// Result and execution state retained for one concrete hierarchy path.
#[derive(Debug)]
pub struct MacroHierarchyNodeResult {
    path: MacroHierarchyPath,
    macro_name: String,
    status: MacroHierarchyNodeStatus,
    result: Option<Arc<MacroExplorationResult>>,
    preview: Option<MacroHierarchyPreviewRecord>,
}

impl MacroHierarchyNodeResult {
    pub(super) fn pending(path: MacroHierarchyPath, macro_name: impl Into<String>) -> Self {
        Self {
            path,
            macro_name: macro_name.into(),
            status: MacroHierarchyNodeStatus::NotReached { blocked_by: None },
            result: None,
            preview: None,
        }
    }

    pub(super) fn set_preview(&mut self, preview: MacroHierarchyPreviewRecord) {
        self.preview = Some(preview);
    }

    pub(super) fn finish(
        &mut self,
        status: MacroHierarchyNodeStatus,
        result: Arc<MacroExplorationResult>,
    ) {
        self.status = status;
        self.result = Some(result);
    }

    pub(super) fn prune(&mut self, reason: MacroDerivationPruneReason) {
        self.status = MacroHierarchyNodeStatus::DerivationPruned { reason };
        self.result = None;
    }

    pub(super) fn finish_blackbox(&mut self, candidates: usize) {
        self.status = MacroHierarchyNodeStatus::BlackBox { candidates };
        self.result = None;
    }

    pub(super) fn block(&mut self, blocked_by: MacroHierarchyPath) {
        if matches!(self.status, MacroHierarchyNodeStatus::NotReached { .. }) {
            self.status = MacroHierarchyNodeStatus::NotReached {
                blocked_by: Some(blocked_by),
            };
        }
    }

    /// Returns the concrete path represented by this record.
    pub const fn path(&self) -> &MacroHierarchyPath {
        &self.path
    }

    /// Returns the reusable macro definition name at this occurrence.
    pub fn macro_name(&self) -> &str {
        &self.macro_name
    }

    /// Returns whether and how this path completed traversal.
    pub const fn status(&self) -> &MacroHierarchyNodeStatus {
        &self.status
    }

    /// Returns the definitive result when this path was evaluated.
    pub fn result(&self) -> Option<&Arc<MacroExplorationResult>> {
        self.result.as_ref()
    }

    /// Returns the provisional parent evaluation, when one was required.
    pub const fn preview(&self) -> Option<&MacroHierarchyPreviewRecord> {
        self.preview.as_ref()
    }
}

/// Parent-to-child derivation trace attached to one concrete child path.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroHierarchyDerivationRecord {
    parent_path: MacroHierarchyPath,
    child_path: MacroHierarchyPath,
    conditions: ResolvedChildConditions,
}

impl MacroHierarchyDerivationRecord {
    pub(super) fn new(
        parent_path: MacroHierarchyPath,
        child_path: MacroHierarchyPath,
        conditions: ResolvedChildConditions,
    ) -> Self {
        Self {
            parent_path,
            child_path,
            conditions,
        }
    }

    /// Returns the macro occurrence whose accepted preview rows were reduced.
    pub const fn parent_path(&self) -> &MacroHierarchyPath {
        &self.parent_path
    }

    /// Returns the direct child occurrence receiving the derived conditions.
    pub const fn child_path(&self) -> &MacroHierarchyPath {
        &self.child_path
    }

    /// Returns effective conditions, rule audit, and optional prune reason.
    pub const fn conditions(&self) -> &ResolvedChildConditions {
        &self.conditions
    }
}

/// Aggregate traversal counts and wall-clock time for one hierarchy run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MacroHierarchyStatistics {
    total_paths: usize,
    explored_leaves: usize,
    refreshed_parents: usize,
    blackboxes: usize,
    finalized_previews: usize,
    preview_rejections: usize,
    derivation_pruned: usize,
    not_reached: usize,
    previews_executed: usize,
    definitive_evaluations: usize,
    frequency_evaluations: usize,
    total_duration: Duration,
}

impl MacroHierarchyStatistics {
    pub(super) fn from_nodes(
        nodes: &BTreeMap<MacroHierarchyPath, MacroHierarchyNodeResult>,
        total_duration: Duration,
    ) -> Self {
        let mut statistics = Self {
            total_paths: nodes.len(),
            total_duration,
            ..Self::default()
        };
        for node in nodes.values() {
            if let Some(preview) = node.preview() {
                statistics.previews_executed += 1;
                statistics.frequency_evaluations += preview.statistics().frequency_evaluations();
            }
            match node.status() {
                MacroHierarchyNodeStatus::NotReached { .. } => statistics.not_reached += 1,
                MacroHierarchyNodeStatus::ExploredLeaf => {
                    statistics.explored_leaves += 1;
                    statistics.definitive_evaluations += 1;
                    statistics.frequency_evaluations += node
                        .result()
                        .map_or(0, |result| result.statistics().frequency_evaluations());
                }
                MacroHierarchyNodeStatus::BlackBox { .. } => statistics.blackboxes += 1,
                MacroHierarchyNodeStatus::PreviewRejected => {
                    statistics.preview_rejections += 1;
                }
                MacroHierarchyNodeStatus::DerivationPruned { .. } => {
                    statistics.derivation_pruned += 1;
                }
                MacroHierarchyNodeStatus::Refreshed => {
                    statistics.refreshed_parents += 1;
                    statistics.definitive_evaluations += 1;
                    statistics.frequency_evaluations += node
                        .result()
                        .map_or(0, |result| result.statistics().frequency_evaluations());
                }
                MacroHierarchyNodeStatus::PreviewFinalized => {
                    statistics.finalized_previews += 1;
                    statistics.definitive_evaluations += 1;
                }
            }
        }
        statistics
    }

    /// Returns every planned macro occurrence, including routes not reached.
    pub const fn total_paths(&self) -> usize {
        self.total_paths
    }
    /// Returns leaf occurrences that completed their definitive evaluation.
    pub const fn explored_leaves(&self) -> usize {
        self.explored_leaves
    }
    /// Returns parents evaluated definitively with projected child candidates.
    pub const fn refreshed_parents(&self) -> usize {
        self.refreshed_parents
    }
    /// Returns child occurrences intentionally treated as compact blackboxes.
    pub const fn blackboxes(&self) -> usize {
        self.blackboxes
    }
    /// Returns parent previews reused directly as definitive results.
    pub const fn finalized_previews(&self) -> usize {
        self.finalized_previews
    }
    /// Returns parent previews that accepted no candidates.
    pub const fn preview_rejections(&self) -> usize {
        self.preview_rejections
    }
    /// Returns child occurrences pruned by contradictory derived conditions.
    pub const fn derivation_pruned(&self) -> usize {
        self.derivation_pruned
    }
    /// Returns descendants not reached because traversal stopped above them.
    pub const fn not_reached(&self) -> usize {
        self.not_reached
    }
    /// Returns provisional parent evaluations executed by the traversal.
    pub const fn previews_executed(&self) -> usize {
        self.previews_executed
    }
    /// Returns leaf evaluations, parent refreshes, and finalized previews.
    pub const fn definitive_evaluations(&self) -> usize {
        self.definitive_evaluations
    }
    /// Returns frequency evaluations accumulated across previews and finals.
    pub const fn frequency_evaluations(&self) -> usize {
        self.frequency_evaluations
    }
    /// Returns wall-clock time for validation and the complete DFS traversal.
    pub const fn total_duration(&self) -> Duration {
        self.total_duration
    }
}

/// Immutable execution tree returned by hierarchical macro exploration.
#[derive(Debug)]
pub struct MacroHierarchyExplorationResult {
    root_path: MacroHierarchyPath,
    root_result: Arc<MacroExplorationResult>,
    nodes: BTreeMap<MacroHierarchyPath, MacroHierarchyNodeResult>,
    derivations: BTreeMap<MacroHierarchyPath, MacroHierarchyDerivationRecord>,
    statistics: MacroHierarchyStatistics,
}

impl MacroHierarchyExplorationResult {
    pub(super) fn new(
        root_path: MacroHierarchyPath,
        root_result: Arc<MacroExplorationResult>,
        nodes: BTreeMap<MacroHierarchyPath, MacroHierarchyNodeResult>,
        derivations: BTreeMap<MacroHierarchyPath, MacroHierarchyDerivationRecord>,
        total_duration: Duration,
    ) -> Self {
        let statistics = MacroHierarchyStatistics::from_nodes(&nodes, total_duration);
        Self {
            root_path,
            root_result,
            nodes,
            derivations,
            statistics,
        }
    }

    /// Returns the path of the requested top-level macro occurrence.
    pub const fn root_path(&self) -> &MacroHierarchyPath {
        &self.root_path
    }

    /// Returns the definitive top-level macro result.
    pub fn root_result(&self) -> &Arc<MacroExplorationResult> {
        &self.root_result
    }

    /// Finds the execution record of one concrete macro occurrence.
    pub fn node(&self, path: &MacroHierarchyPath) -> Option<&MacroHierarchyNodeResult> {
        self.nodes.get(path)
    }

    /// Iterates through all planned routes in deterministic path order.
    pub fn nodes(&self) -> impl Iterator<Item = (&MacroHierarchyPath, &MacroHierarchyNodeResult)> {
        self.nodes.iter()
    }

    /// Returns the parent-derived conditions recorded for one child path.
    pub fn derivation(
        &self,
        child_path: &MacroHierarchyPath,
    ) -> Option<&MacroHierarchyDerivationRecord> {
        self.derivations.get(child_path)
    }

    /// Iterates through derivation records in deterministic child-path order.
    pub fn derivations(
        &self,
    ) -> impl Iterator<Item = (&MacroHierarchyPath, &MacroHierarchyDerivationRecord)> {
        self.derivations.iter()
    }

    /// Returns aggregate traversal counts and wall-clock duration.
    pub const fn statistics(&self) -> &MacroHierarchyStatistics {
        &self.statistics
    }
}
