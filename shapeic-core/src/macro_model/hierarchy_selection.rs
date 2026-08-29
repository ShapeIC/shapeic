//! Selection-specific traversal of recursive macro candidate provenance.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::analysis::AdaptiveAcOutcome;
use crate::exploration::candidate::CandidatePoint;

use super::{
    MacroAcceptedCandidate, MacroExplorationInstanceKind, MacroExplorationResult,
    MacroHierarchyExplorationResult, MacroHierarchyPath, MacroHierarchyPathError,
    MacroInstanceCandidateSet,
};

/// Typed path of one local instance inside a concrete macro occurrence.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct MacroHierarchyInstancePath {
    macro_path: MacroHierarchyPath,
    instance: String,
}

impl MacroHierarchyInstancePath {
    /// Creates a path from its owning macro occurrence and local instance name.
    pub fn new(
        macro_path: MacroHierarchyPath,
        instance: impl Into<String>,
    ) -> Result<Self, MacroHierarchyInstancePathError> {
        let instance = instance.into();
        if instance.trim().is_empty() {
            return Err(MacroHierarchyInstancePathError::EmptyInstance);
        }
        if instance.contains("::") {
            return Err(MacroHierarchyInstancePathError::AmbiguousInstance { instance });
        }
        Ok(Self {
            macro_path,
            instance,
        })
    }

    /// Returns the macro occurrence owning the instance.
    pub const fn macro_path(&self) -> &MacroHierarchyPath {
        &self.macro_path
    }

    /// Returns the instance name local to its owning macro.
    pub fn instance(&self) -> &str {
        &self.instance
    }
}

impl fmt::Display for MacroHierarchyInstancePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}::{}", self.macro_path, self.instance)
    }
}

/// Invalid local-instance component in a hierarchical instance path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroHierarchyInstancePathError {
    /// The local instance name is empty.
    EmptyInstance,
    /// The local instance contains the display separator.
    AmbiguousInstance { instance: String },
}

impl fmt::Display for MacroHierarchyInstancePathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInstance => formatter.write_str("hierarchical instance name is empty"),
            Self::AmbiguousInstance { instance } => write!(
                formatter,
                "hierarchical instance name '{instance}' contains the reserved '::' separator"
            ),
        }
    }
}

impl Error for MacroHierarchyInstancePathError {}

/// One complete implementation choice rooted at an accepted top-level row.
#[derive(Debug)]
pub struct MacroHierarchySelection<'result> {
    hierarchy: &'result MacroHierarchyExplorationResult,
    root_accepted_index: usize,
    selected_nodes: BTreeMap<MacroHierarchyPath, usize>,
    traversal_order: Vec<MacroHierarchyPath>,
}

impl MacroHierarchyExplorationResult {
    /// Resolves one accepted root row through all recursive child provenance.
    pub fn selection(
        &self,
        root_accepted_index: usize,
    ) -> Result<MacroHierarchySelection<'_>, MacroHierarchySelectionError> {
        MacroHierarchySelection::new(self, root_accepted_index)
    }
}

impl<'result> MacroHierarchySelection<'result> {
    fn new(
        hierarchy: &'result MacroHierarchyExplorationResult,
        root_accepted_index: usize,
    ) -> Result<Self, MacroHierarchySelectionError> {
        if hierarchy
            .root_result()
            .accepted()
            .get(root_accepted_index)
            .is_none()
        {
            return Err(MacroHierarchySelectionError::RootAcceptedIndexOutOfRange {
                index: root_accepted_index,
                accepted_count: hierarchy.root_result().accepted().len(),
            });
        }
        let mut selection = Self {
            hierarchy,
            root_accepted_index,
            selected_nodes: BTreeMap::new(),
            traversal_order: Vec::new(),
        };
        selection.visit(
            hierarchy.root_path().clone(),
            hierarchy.root_result(),
            root_accepted_index,
        )?;
        Ok(selection)
    }

    fn visit(
        &mut self,
        path: MacroHierarchyPath,
        result: &Arc<MacroExplorationResult>,
        accepted_index: usize,
    ) -> Result<(), MacroHierarchySelectionError> {
        if self.selected_nodes.contains_key(&path) {
            return Err(MacroHierarchySelectionError::DuplicateMacroPath { path });
        }
        let registered = self
            .hierarchy
            .node(&path)
            .ok_or_else(|| MacroHierarchySelectionError::UnknownMacroPath { path: path.clone() })?;
        let registered_result = registered.result().ok_or_else(|| {
            MacroHierarchySelectionError::MissingFinalResult { path: path.clone() }
        })?;
        if !Arc::ptr_eq(registered_result, result) {
            return Err(MacroHierarchySelectionError::ResultProvenanceMismatch {
                path: path.clone(),
            });
        }
        let accepted = result.accepted().get(accepted_index).ok_or_else(|| {
            MacroHierarchySelectionError::AcceptedIndexOutOfRange {
                path: path.clone(),
                index: accepted_index,
                accepted_count: result.accepted().len(),
            }
        })?;
        self.selected_nodes.insert(path.clone(), accepted_index);
        self.traversal_order.push(path.clone());

        for instance in result.candidate_sets().instances() {
            let candidate_index = result
                .selected_candidate_index(accepted, instance.instance_path())
                .ok_or_else(|| MacroHierarchySelectionError::MissingCandidateIndex {
                    path: path.clone(),
                    instance: instance.instance_path().to_owned(),
                })?;
            if instance.candidates().points.get(candidate_index).is_none() {
                return Err(MacroHierarchySelectionError::CandidateIndexOutOfRange {
                    path: path.clone(),
                    instance: instance.instance_path().to_owned(),
                    index: candidate_index,
                    candidate_count: instance.candidates().points.len(),
                });
            }
            if instance.kind() != MacroExplorationInstanceKind::CompactMacro {
                continue;
            }
            let child_path = path.child(instance.instance_path()).map_err(|error| {
                MacroHierarchySelectionError::InvalidChildPath {
                    path: path.clone(),
                    instance: instance.instance_path().to_owned(),
                    error,
                }
            })?;
            let (child_result, child_accepted_index, _) = result
                .selected_submacro_with_index(accepted, instance.instance_path())
                .ok_or_else(|| MacroHierarchySelectionError::MissingSubmacroProvenance {
                    path: path.clone(),
                    instance: instance.instance_path().to_owned(),
                })?;
            self.visit(child_path, child_result, child_accepted_index)?;
        }
        Ok(())
    }

    /// Returns the accepted-row index requested at the root.
    pub const fn root_accepted_index(&self) -> usize {
        self.root_accepted_index
    }

    /// Returns the selected accepted row for one concrete macro occurrence.
    pub fn node(&self, path: &MacroHierarchyPath) -> Option<MacroHierarchySelectedNode<'_>> {
        let (selected_path, accepted_index) = self.selected_nodes.get_key_value(path)?;
        let accepted_index = *accepted_index;
        let result = self.hierarchy.node(selected_path)?.result()?.as_ref();
        let accepted = result.accepted().get(accepted_index)?;
        Some(MacroHierarchySelectedNode {
            path: selected_path,
            result,
            accepted_index,
            accepted,
        })
    }

    /// Iterates selected macro occurrences in deterministic depth-first order.
    pub fn nodes(&self) -> impl Iterator<Item = MacroHierarchySelectedNode<'_>> {
        self.traversal_order
            .iter()
            .filter_map(|path| self.node(path))
    }

    /// Returns the selected candidate of one local primitive or compact macro.
    pub fn instance(
        &self,
        path: &MacroHierarchyInstancePath,
    ) -> Option<MacroHierarchySelectedInstance<'_>> {
        let node = self.node(path.macro_path())?;
        selected_instance(node, path.instance())
    }

    /// Iterates every selected local instance in macro DFS and circuit order.
    pub fn instances(&self) -> impl Iterator<Item = MacroHierarchySelectedInstance<'_>> {
        self.nodes().flat_map(|node| {
            node.result
                .candidate_sets()
                .instances()
                .iter()
                .filter_map(move |instance| selected_instance(node, instance.instance_path()))
        })
    }

    /// Iterates only selected primitive instances, excluding compact submacros.
    pub fn primitive_instances(&self) -> impl Iterator<Item = MacroHierarchySelectedInstance<'_>> {
        self.instances()
            .filter(|instance| instance.kind() == MacroExplorationInstanceKind::Primitive)
    }
}

/// Selected accepted row of one macro occurrence.
#[derive(Clone, Copy, Debug)]
pub struct MacroHierarchySelectedNode<'result> {
    path: &'result MacroHierarchyPath,
    result: &'result MacroExplorationResult,
    accepted_index: usize,
    accepted: &'result MacroAcceptedCandidate,
}

impl<'result> MacroHierarchySelectedNode<'result> {
    /// Returns the concrete macro occurrence.
    pub const fn path(&self) -> &'result MacroHierarchyPath {
        self.path
    }
    /// Returns the reusable macro definition name.
    pub fn macro_name(&self) -> &'result str {
        self.result.macro_name()
    }
    /// Returns the selected accepted-row index.
    pub const fn accepted_index(&self) -> usize {
        self.accepted_index
    }
    /// Returns the complete immutable result of this macro occurrence.
    pub const fn result(&self) -> &'result MacroExplorationResult {
        self.result
    }
    /// Returns the selected accepted row.
    pub const fn accepted_candidate(&self) -> &'result MacroAcceptedCandidate {
        self.accepted
    }
    /// Resolves one analysis-independent specification value.
    pub fn specification_value(&self, name: &str) -> Option<f64> {
        self.accepted.specification_value(name)
    }
    /// Resolves one AC outcome by macro-local testbench name.
    pub fn ac_outcome(&self, testbench: &str) -> Option<&'result AdaptiveAcOutcome> {
        self.result.ac_outcome(self.accepted, testbench)
    }
}

/// Selected candidate point of one local primitive or compact submacro.
#[derive(Clone, Copy, Debug)]
pub struct MacroHierarchySelectedInstance<'result> {
    macro_path: &'result MacroHierarchyPath,
    instance: &'result MacroInstanceCandidateSet,
    candidate_index: usize,
    candidate: &'result CandidatePoint,
}

impl<'result> MacroHierarchySelectedInstance<'result> {
    /// Returns the macro occurrence owning this instance.
    pub const fn macro_path(&self) -> &'result MacroHierarchyPath {
        self.macro_path
    }
    /// Returns the instance name local to the owning macro.
    pub fn instance(&self) -> &'result str {
        self.instance.instance_path()
    }
    /// Returns whether this is a primitive or compact submacro candidate.
    pub const fn kind(&self) -> MacroExplorationInstanceKind {
        self.instance.kind()
    }
    /// Returns the selected index in the local candidate set.
    pub const fn candidate_index(&self) -> usize {
        self.candidate_index
    }
    /// Returns the selected candidate without copying its columns.
    pub const fn candidate_point(&self) -> &'result CandidatePoint {
        self.candidate
    }
    /// Resolves one exact candidate column.
    pub fn value(&self, column: &str) -> Option<f64> {
        self.candidate.get(column)
    }
    /// Creates the typed full instance path on demand.
    pub fn path(&self) -> MacroHierarchyInstancePath {
        MacroHierarchyInstancePath::new(self.macro_path.clone(), self.instance())
            .expect("validated circuit instance names produce typed paths")
    }
}

fn selected_instance<'result>(
    node: MacroHierarchySelectedNode<'result>,
    instance_path: &str,
) -> Option<MacroHierarchySelectedInstance<'result>> {
    let instance = node.result.candidate_sets().instance(instance_path)?;
    let candidate_index = node
        .result
        .selected_candidate_index(node.accepted, instance_path)?;
    let candidate = instance.candidates().points.get(candidate_index)?;
    Some(MacroHierarchySelectedInstance {
        macro_path: node.path,
        instance,
        candidate_index,
        candidate,
    })
}

/// Broken or inconsistent recursive candidate provenance.
#[derive(Debug)]
pub enum MacroHierarchySelectionError {
    /// The requested root row does not exist.
    RootAcceptedIndexOutOfRange {
        /// Requested accepted-row index.
        index: usize,
        /// Number of accepted root rows.
        accepted_count: usize,
    },
    /// Recursive provenance points to a path absent from the execution tree.
    UnknownMacroPath {
        /// Missing macro occurrence.
        path: MacroHierarchyPath,
    },
    /// A selected path exists but did not retain a definitive result.
    MissingFinalResult {
        /// Macro occurrence without a definitive result.
        path: MacroHierarchyPath,
    },
    /// Recursive provenance and the execution tree retain different results.
    ResultProvenanceMismatch {
        /// Macro occurrence whose result identity differs.
        path: MacroHierarchyPath,
    },
    /// Recursive provenance selects an accepted row that does not exist.
    AcceptedIndexOutOfRange {
        /// Macro occurrence containing the invalid selection.
        path: MacroHierarchyPath,
        /// Selected accepted-row index.
        index: usize,
        /// Number of accepted rows in the result.
        accepted_count: usize,
    },
    /// An accepted row does not select a candidate for one local instance.
    MissingCandidateIndex {
        /// Macro occurrence containing the instance.
        path: MacroHierarchyPath,
        /// Local instance name.
        instance: String,
    },
    /// An accepted row selects a candidate index outside its local set.
    CandidateIndexOutOfRange {
        /// Macro occurrence containing the instance.
        path: MacroHierarchyPath,
        /// Local instance name.
        instance: String,
        /// Selected local candidate index.
        index: usize,
        /// Number of candidates in the local set.
        candidate_count: usize,
    },
    /// A compact instance name cannot form a valid child hierarchy path.
    InvalidChildPath {
        /// Parent macro occurrence.
        path: MacroHierarchyPath,
        /// Invalid local child-instance name.
        instance: String,
        /// Typed path validation error.
        error: MacroHierarchyPathError,
    },
    /// A selected compact candidate has no child-result provenance.
    MissingSubmacroProvenance {
        /// Parent macro occurrence.
        path: MacroHierarchyPath,
        /// Local compact instance name.
        instance: String,
    },
    /// Recursive traversal reached the same macro occurrence twice.
    DuplicateMacroPath {
        /// Duplicated macro occurrence.
        path: MacroHierarchyPath,
    },
}

impl fmt::Display for MacroHierarchySelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootAcceptedIndexOutOfRange {
                index,
                accepted_count,
            } => write!(
                formatter,
                "root accepted-row index {index} is outside 0..{accepted_count}"
            ),
            Self::UnknownMacroPath { path } => {
                write!(
                    formatter,
                    "selected macro path '{path}' is absent from the hierarchy"
                )
            }
            Self::MissingFinalResult { path } => write!(
                formatter,
                "selected macro path '{path}' has no definitive exploration result"
            ),
            Self::ResultProvenanceMismatch { path } => write!(
                formatter,
                "selected macro path '{path}' refers to a different result than its recursive provenance"
            ),
            Self::AcceptedIndexOutOfRange {
                path,
                index,
                accepted_count,
            } => write!(
                formatter,
                "selected accepted-row index {index} for '{path}' is outside 0..{accepted_count}"
            ),
            Self::MissingCandidateIndex { path, instance } => write!(
                formatter,
                "selected row for '{path}' has no candidate index for instance '{instance}'"
            ),
            Self::CandidateIndexOutOfRange {
                path,
                instance,
                index,
                candidate_count,
            } => write!(
                formatter,
                "selected candidate index {index} for '{path}::{instance}' is outside 0..{candidate_count}"
            ),
            Self::InvalidChildPath { path, instance, .. } => write!(
                formatter,
                "instance '{instance}' below '{path}' cannot form a hierarchy path"
            ),
            Self::MissingSubmacroProvenance { path, instance } => write!(
                formatter,
                "selected compact instance '{path}::{instance}' has no child-result provenance"
            ),
            Self::DuplicateMacroPath { path } => write!(
                formatter,
                "selected recursive provenance reaches macro path '{path}' more than once"
            ),
        }
    }
}
impl Error for MacroHierarchySelectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidChildPath { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ptr;
    use std::time::Duration;

    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::exploration::filter::CandidateFilterReport;

    use super::super::input::CompactMacroCandidateProvenance;
    use super::super::{
        MacroCandidateSets, MacroExplorationInstanceKind, MacroHierarchyNodeResult,
        MacroHierarchyNodeStatus,
    };
    use super::*;

    fn local_result(column: &str, value: f64) -> Arc<MacroExplorationResult> {
        Arc::new(MacroExplorationResult::test_fixture(
            "root",
            MacroCandidateSets {
                instances: vec![MacroInstanceCandidateSet {
                    instance_path: "xm1".to_owned(),
                    kind: MacroExplorationInstanceKind::Primitive,
                    candidates: CandidateSet::new(
                        "xm1",
                        vec![CandidatePoint::new(vec![(column.to_owned(), value)])],
                    ),
                    filter_report: CandidateFilterReport::default(),
                    interface_ports: Vec::new(),
                    compact_provenance: None,
                }],
            },
            vec![(vec![0], vec![("gain".to_owned(), 42.0)])],
        ))
    }

    fn hierarchy_with_root(
        root_result: Arc<MacroExplorationResult>,
    ) -> MacroHierarchyExplorationResult {
        let root_path = MacroHierarchyPath::root("root").unwrap();
        let mut root_node = MacroHierarchyNodeResult::pending(root_path.clone(), "root");
        root_node.finish(
            MacroHierarchyNodeStatus::ExploredLeaf,
            Arc::clone(&root_result),
        );
        MacroHierarchyExplorationResult::new(
            root_path.clone(),
            root_result,
            BTreeMap::from([(root_path, root_node)]),
            BTreeMap::new(),
            Duration::ZERO,
        )
    }

    #[test]
    fn exposes_exact_local_candidate_without_copying_it() {
        let result = hierarchy_with_root(local_result("width__xm1__m1", 10.0));
        let selection = result.selection(0).unwrap();
        let path = MacroHierarchyInstancePath::new(result.root_path().clone(), "xm1").unwrap();
        let instance = selection.instance(&path).unwrap();
        let stored = &result.root_result().candidate_sets().instances()[0]
            .candidates()
            .points[0];

        assert_eq!(path.to_string(), "root::xm1");
        assert_eq!(instance.value("width__xm1__m1"), Some(10.0));
        assert_eq!(instance.candidate_index(), 0);
        assert!(ptr::eq(instance.candidate_point(), stored));
        assert_eq!(
            selection
                .nodes()
                .next()
                .unwrap()
                .specification_value("gain"),
            Some(42.0)
        );
        assert_eq!(selection.primitive_instances().count(), 1);
    }

    #[test]
    fn rejects_a_root_row_outside_the_accepted_results() {
        let result = hierarchy_with_root(local_result("length__xm1__m1", 0.4));
        assert!(matches!(
            result.selection(1),
            Err(MacroHierarchySelectionError::RootAcceptedIndexOutOfRange {
                index: 1,
                accepted_count: 1
            })
        ));
    }

    #[test]
    fn rejects_compact_candidates_without_recursive_provenance() {
        let root_path = MacroHierarchyPath::root("root").unwrap();
        let child_path = root_path.child("xchild").unwrap();
        let root_result = Arc::new(MacroExplorationResult::test_fixture(
            "root",
            MacroCandidateSets {
                instances: vec![MacroInstanceCandidateSet {
                    instance_path: "xchild".to_owned(),
                    kind: MacroExplorationInstanceKind::CompactMacro,
                    candidates: CandidateSet::new("xchild", vec![CandidatePoint::new(Vec::new())]),
                    filter_report: CandidateFilterReport::default(),
                    interface_ports: Vec::new(),
                    compact_provenance: None,
                }],
            },
            vec![(vec![0], Vec::new())],
        ));
        let mut root_node = MacroHierarchyNodeResult::pending(root_path.clone(), "root");
        root_node.finish(
            MacroHierarchyNodeStatus::Refreshed,
            Arc::clone(&root_result),
        );
        let nodes = BTreeMap::from([
            (root_path.clone(), root_node),
            (
                child_path.clone(),
                MacroHierarchyNodeResult::pending(child_path, "child"),
            ),
        ]);
        let hierarchy = MacroHierarchyExplorationResult::new(
            root_path,
            root_result,
            nodes,
            BTreeMap::new(),
            Duration::ZERO,
        );

        assert!(matches!(
            hierarchy.selection(0),
            Err(MacroHierarchySelectionError::MissingSubmacroProvenance { instance, .. })
                if instance == "xchild"
        ));
    }

    #[test]
    fn rejects_a_child_result_that_differs_from_the_registered_tree() {
        let root_path = MacroHierarchyPath::root("root").unwrap();
        let child_path = root_path.child("xchild").unwrap();
        let provenance_child = local_result("width", 1.0);
        let registered_child = local_result("width", 2.0);
        let root_result = Arc::new(MacroExplorationResult::test_fixture(
            "root",
            MacroCandidateSets {
                instances: vec![MacroInstanceCandidateSet {
                    instance_path: "xchild".to_owned(),
                    kind: MacroExplorationInstanceKind::CompactMacro,
                    candidates: CandidateSet::new("xchild", vec![CandidatePoint::new(Vec::new())]),
                    filter_report: CandidateFilterReport::default(),
                    interface_ports: Vec::new(),
                    compact_provenance: Some(CompactMacroCandidateProvenance {
                        source_result: provenance_child,
                        accepted_indices: vec![0],
                    }),
                }],
            },
            vec![(vec![0], Vec::new())],
        ));
        let mut root_node = MacroHierarchyNodeResult::pending(root_path.clone(), "root");
        root_node.finish(
            MacroHierarchyNodeStatus::Refreshed,
            Arc::clone(&root_result),
        );
        let mut child_node = MacroHierarchyNodeResult::pending(child_path.clone(), "child");
        child_node.finish(MacroHierarchyNodeStatus::ExploredLeaf, registered_child);
        let hierarchy = MacroHierarchyExplorationResult::new(
            root_path.clone(),
            root_result,
            BTreeMap::from([(root_path, root_node), (child_path.clone(), child_node)]),
            BTreeMap::new(),
            Duration::ZERO,
        );

        assert!(matches!(
            hierarchy.selection(0),
            Err(MacroHierarchySelectionError::ResultProvenanceMismatch { path })
                if path == child_path
        ));
    }

    #[test]
    fn validates_typed_instance_components() {
        let root = MacroHierarchyPath::root("root").unwrap();
        assert!(matches!(
            MacroHierarchyInstancePath::new(root.clone(), ""),
            Err(MacroHierarchyInstancePathError::EmptyInstance)
        ));
        assert!(matches!(
            MacroHierarchyInstancePath::new(root, "xa::xm1"),
            Err(MacroHierarchyInstancePathError::AmbiguousInstance { .. })
        ));
    }
}
