//! One-pass electrical preview, recursive child exploration, and parent refresh.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::BlockRef;

use super::{
    CompactMacroInstanceExplorationInput, Macro, MacroCandidateProjectionError, MacroCatalog,
    MacroDerivationResolutionError, MacroExplorationError, MacroExplorationResult,
    MacroHierarchyDerivationRecord, MacroHierarchyExplorationInput,
    MacroHierarchyExplorationResult, MacroHierarchyLocalInputError, MacroHierarchyNodeResult,
    MacroHierarchyNodeStatus, MacroHierarchyPath, MacroHierarchyPathError,
    MacroHierarchyPreviewRecord, MacroHierarchyValidationError, ResolvedChildConditions,
    resolve_child_derivations, validate_macro_hierarchy_input,
};

/// Stage of one path-local exploration that failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroHierarchyExplorationStage {
    /// Provisional parent evaluation using nominal child seeds.
    Preview,
    /// Single definitive evaluation of a macro without submacros.
    Leaf,
    /// Definitive parent evaluation using explored child projections.
    Refresh,
}

impl fmt::Display for MacroHierarchyExplorationStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preview => formatter.write_str("preview"),
            Self::Leaf => formatter.write_str("leaf exploration"),
            Self::Refresh => formatter.write_str("parent refresh"),
        }
    }
}

/// Runs a deterministic, one-pass electrical depth-first macro exploration.
///
/// Parent previews use nominal compact seeds. Accepted preview rows derive the
/// conditions of each direct child. Explored children are then projected into
/// the parent, which is evaluated once more to produce the returned result.
pub fn explore_macro_hierarchy(
    top_macro: &str,
    macro_catalog: &MacroCatalog,
    primitive_catalog: &PrimitiveCatalog,
    input: &MacroHierarchyExplorationInput<'_>,
) -> Result<MacroHierarchyExplorationResult, MacroHierarchyExplorationError> {
    let start = Instant::now();
    let validation_errors =
        validate_macro_hierarchy_input(top_macro, macro_catalog, primitive_catalog, input);
    if !validation_errors.is_empty() {
        return Err(MacroHierarchyExplorationError::Validation {
            errors: validation_errors,
        });
    }
    let macro_ =
        macro_catalog
            .get(top_macro)
            .ok_or_else(|| MacroHierarchyExplorationError::Validation {
                errors: vec![MacroHierarchyValidationError::UnknownTopMacro {
                    macro_name: top_macro.to_owned(),
                }],
            })?;
    let path = MacroHierarchyPath::root(top_macro).map_err(|error| {
        MacroHierarchyExplorationError::Path {
            parent_path: None,
            child_instance: top_macro.to_owned(),
            error,
        }
    })?;
    let mut nodes = BTreeMap::new();
    collect_planned_nodes(macro_, macro_catalog, path.clone(), &mut nodes);
    let mut derivations = BTreeMap::new();
    let root_result = explore_node(
        macro_,
        path.clone(),
        None,
        macro_catalog,
        primitive_catalog,
        input,
        &mut nodes,
        &mut derivations,
    )?;
    Ok(MacroHierarchyExplorationResult::new(
        path,
        root_result,
        nodes,
        derivations,
        start.elapsed(),
    ))
}

fn explore_node(
    macro_: &Macro,
    path: MacroHierarchyPath,
    inherited: Option<&ResolvedChildConditions>,
    macro_catalog: &MacroCatalog,
    primitive_catalog: &PrimitiveCatalog,
    hierarchy_input: &MacroHierarchyExplorationInput<'_>,
    nodes: &mut BTreeMap<MacroHierarchyPath, MacroHierarchyNodeResult>,
    derivations: &mut BTreeMap<MacroHierarchyPath, MacroHierarchyDerivationRecord>,
) -> Result<Arc<MacroExplorationResult>, MacroHierarchyExplorationError> {
    let children = macro_
        .circuit()
        .instances()
        .iter()
        .filter_map(|instance| match instance.block() {
            BlockRef::Macro(child_macro) => Some((instance.name(), child_macro.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    let initial_input = hierarchy_input
        .local_input(macro_, macro_catalog, &path, inherited, None)
        .map_err(|errors| MacroHierarchyExplorationError::LocalInput {
            path: path.clone(),
            macro_name: macro_.name().to_owned(),
            errors,
        })?;

    if children.is_empty() {
        let result = macro_
            .explore(primitive_catalog, macro_catalog, initial_input)
            .map_err(|error| MacroHierarchyExplorationError::Explore {
                path: path.clone(),
                macro_name: macro_.name().to_owned(),
                stage: MacroHierarchyExplorationStage::Leaf,
                error,
            })?;
        let result = Arc::new(result);
        node_mut(nodes, &path).finish(MacroHierarchyNodeStatus::ExploredLeaf, Arc::clone(&result));
        return Ok(result);
    }

    let preview = Arc::new(
        macro_
            .explore(primitive_catalog, macro_catalog, initial_input)
            .map_err(|error| MacroHierarchyExplorationError::Explore {
                path: path.clone(),
                macro_name: macro_.name().to_owned(),
                stage: MacroHierarchyExplorationStage::Preview,
                error,
            })?,
    );
    node_mut(nodes, &path).set_preview(MacroHierarchyPreviewRecord::new(
        &preview,
        hierarchy_input.retention_policy(),
    ));
    if preview.accepted().is_empty() {
        node_mut(nodes, &path).finish(
            MacroHierarchyNodeStatus::PreviewRejected,
            Arc::clone(&preview),
        );
        block_descendants(nodes, &path, &path);
        return Ok(preview);
    }

    let mut resolved_children = BTreeMap::new();
    for (child_instance, child_macro_name) in children {
        let child_macro = macro_catalog.get(child_macro_name).ok_or_else(|| {
            MacroHierarchyExplorationError::MissingChildMacro {
                path: path.clone(),
                parent_macro: macro_.name().to_owned(),
                child_instance: child_instance.to_owned(),
                child_macro: child_macro_name.to_owned(),
            }
        })?;
        let child_path =
            path.child(child_instance)
                .map_err(|error| MacroHierarchyExplorationError::Path {
                    parent_path: Some(path.clone()),
                    child_instance: child_instance.to_owned(),
                    error,
                })?;
        let conditions =
            resolve_child_derivations(macro_, &preview, child_instance).map_err(|error| {
                MacroHierarchyExplorationError::Derivation {
                    path: path.clone(),
                    macro_name: macro_.name().to_owned(),
                    child_instance: child_instance.to_owned(),
                    error,
                }
            })?;
        derivations.insert(
            child_path.clone(),
            MacroHierarchyDerivationRecord::new(
                path.clone(),
                child_path.clone(),
                conditions.clone(),
            ),
        );
        let compact_input = if conditions.is_pruned() {
            let reason = conditions
                .prune_reason()
                .expect("pruned conditions always retain their reason")
                .clone();
            node_mut(nodes, &child_path).prune(reason);
            block_descendants(nodes, &child_path, &child_path);
            CompactMacroInstanceExplorationInput::empty_for_pruned_child(
                child_macro,
                child_instance.to_owned(),
            )
        } else {
            let child_result = explore_node(
                child_macro,
                child_path.clone(),
                Some(&conditions),
                macro_catalog,
                primitive_catalog,
                hierarchy_input,
                nodes,
                derivations,
            )?;
            let projection =
                child_result
                    .project(child_macro, child_instance)
                    .map_err(|error| MacroHierarchyExplorationError::Projection {
                        parent_path: path.clone(),
                        child_path,
                        child_instance: child_instance.to_owned(),
                        child_macro: child_macro.name().to_owned(),
                        error,
                    })?;
            CompactMacroInstanceExplorationInput::from_projection(projection, Vec::new())
        };
        resolved_children.insert(child_instance.to_owned(), compact_input);
    }

    let final_input = hierarchy_input
        .local_input(
            macro_,
            macro_catalog,
            &path,
            inherited,
            Some(&resolved_children),
        )
        .map_err(|errors| MacroHierarchyExplorationError::LocalInput {
            path: path.clone(),
            macro_name: macro_.name().to_owned(),
            errors,
        })?;
    let result = macro_
        .explore(primitive_catalog, macro_catalog, final_input)
        .map_err(|error| MacroHierarchyExplorationError::Explore {
            path: path.clone(),
            macro_name: macro_.name().to_owned(),
            stage: MacroHierarchyExplorationStage::Refresh,
            error,
        })?;
    let result = Arc::new(result);
    node_mut(nodes, &path).finish(MacroHierarchyNodeStatus::Refreshed, Arc::clone(&result));
    Ok(result)
}

fn node_mut<'a>(
    nodes: &'a mut BTreeMap<MacroHierarchyPath, MacroHierarchyNodeResult>,
    path: &MacroHierarchyPath,
) -> &'a mut MacroHierarchyNodeResult {
    nodes
        .get_mut(path)
        .expect("validated reachable paths are initialized before exploration")
}

fn block_descendants(
    nodes: &mut BTreeMap<MacroHierarchyPath, MacroHierarchyNodeResult>,
    ancestor: &MacroHierarchyPath,
    blocked_by: &MacroHierarchyPath,
) {
    for (path, node) in nodes.iter_mut() {
        if path.components().len() > ancestor.components().len()
            && path.components().starts_with(ancestor.components())
        {
            node.block(blocked_by.clone());
        }
    }
}

fn collect_planned_nodes(
    macro_: &Macro,
    macro_catalog: &MacroCatalog,
    path: MacroHierarchyPath,
    nodes: &mut BTreeMap<MacroHierarchyPath, MacroHierarchyNodeResult>,
) {
    nodes.insert(
        path.clone(),
        MacroHierarchyNodeResult::pending(path.clone(), macro_.name()),
    );
    for instance in macro_.circuit().instances() {
        let BlockRef::Macro(child_macro_name) = instance.block() else {
            continue;
        };
        let (Some(child_macro), Ok(child_path)) = (
            macro_catalog.get(child_macro_name),
            path.child(instance.name()),
        ) else {
            continue;
        };
        collect_planned_nodes(child_macro, macro_catalog, child_path, nodes);
    }
}

/// Path-qualified failure produced by hierarchical exploration.
#[derive(Debug)]
pub enum MacroHierarchyExplorationError {
    /// Static configuration is invalid, so no preview was executed.
    Validation {
        errors: Vec<MacroHierarchyValidationError>,
    },
    /// A local direct-exploration input could not be assembled.
    LocalInput {
        path: MacroHierarchyPath,
        macro_name: String,
        errors: Vec<MacroHierarchyLocalInputError>,
    },
    /// One path-local preview, leaf evaluation, or refresh failed.
    Explore {
        path: MacroHierarchyPath,
        macro_name: String,
        stage: MacroHierarchyExplorationStage,
        error: MacroExplorationError,
    },
    /// A parent derivation rule failed for one direct child.
    Derivation {
        path: MacroHierarchyPath,
        macro_name: String,
        child_instance: String,
        error: MacroDerivationResolutionError,
    },
    /// An accepted child result could not be projected into its parent.
    Projection {
        parent_path: MacroHierarchyPath,
        child_path: MacroHierarchyPath,
        child_instance: String,
        child_macro: String,
        error: MacroCandidateProjectionError,
    },
    /// An invalid component prevented construction of a child path.
    Path {
        parent_path: Option<MacroHierarchyPath>,
        child_instance: String,
        error: MacroHierarchyPathError,
    },
    /// A child macro disappeared after successful hierarchy validation.
    MissingChildMacro {
        path: MacroHierarchyPath,
        parent_macro: String,
        child_instance: String,
        child_macro: String,
    },
}

impl fmt::Display for MacroHierarchyExplorationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation { errors } => write!(
                formatter,
                "hierarchical exploration has {} validation error(s)",
                errors.len()
            ),
            Self::LocalInput {
                path,
                macro_name,
                errors,
            } => write!(
                formatter,
                "could not assemble input for path '{path}' (macro '{macro_name}'): {} error(s)",
                errors.len()
            ),
            Self::Explore {
                path,
                macro_name,
                stage,
                error,
            } => write!(
                formatter,
                "{stage} failed at path '{path}' (macro '{macro_name}'): {error}"
            ),
            Self::Derivation {
                path,
                macro_name,
                child_instance,
                error,
            } => write!(
                formatter,
                "derivation failed at path '{path}' (macro '{macro_name}') for child '{child_instance}': {error}"
            ),
            Self::Projection {
                parent_path,
                child_path,
                child_instance,
                child_macro,
                error,
            } => write!(
                formatter,
                "could not project path '{child_path}' (macro '{child_macro}') as child '{child_instance}' of '{parent_path}': {error}"
            ),
            Self::Path {
                parent_path,
                child_instance,
                error,
            } => write!(
                formatter,
                "could not append hierarchy component '{child_instance}' to '{}': {error}",
                parent_path
                    .as_ref()
                    .map_or_else(|| "<root>".to_owned(), ToString::to_string)
            ),
            Self::MissingChildMacro {
                path,
                parent_macro,
                child_instance,
                child_macro,
            } => write!(
                formatter,
                "path '{path}' (macro '{parent_macro}') child '{child_instance}' references missing macro '{child_macro}'"
            ),
        }
    }
}

impl Error for MacroHierarchyExplorationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Explore { error, .. } => Some(error),
            Self::Derivation { error, .. } => Some(error),
            Self::Projection { error, .. } => Some(error),
            Self::Path { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use crate::circuit::Circuit;

    use super::super::{
        MacroCompactSeed, MacroDerivationReduction, MacroDerivationRule, MacroDerivationTarget,
        MacroSpecification, MacroSpecificationBounds, MacroSpecificationSource,
    };
    use super::*;

    fn leaf(name: &str) -> Macro {
        Macro::new(
            name,
            Vec::new(),
            Circuit::builder().resistor("ri", "a", "b", 1.0).build(),
            Circuit::builder().resistor("rc", "a", "b", 1.0).build(),
        )
        .with_compact_seed(MacroCompactSeed::default())
    }

    fn parent(children: &[(&str, &str)]) -> Macro {
        let mut implementation = Circuit::builder();
        for (instance, macro_name) in children {
            implementation =
                implementation.macro_instance(*instance, *macro_name, Vec::<(&str, &str)>::new());
        }
        Macro::new(
            "top",
            Vec::new(),
            implementation.build(),
            Circuit::builder().resistor("rc", "a", "b", 1.0).build(),
        )
    }

    #[test]
    fn explores_repeated_children_independently_and_preserves_both_provenances() {
        let catalog =
            MacroCatalog::from_macros([leaf("stage"), parent(&[("xa", "stage"), ("xb", "stage")])])
                .unwrap();
        let result = explore_macro_hierarchy(
            "top",
            &catalog,
            &PrimitiveCatalog::new(),
            &MacroHierarchyExplorationInput::new(),
        )
        .unwrap();
        let root = result.root_result();
        let accepted = &root.accepted()[0];
        let (xa, _) = root.selected_submacro(accepted, "xa").unwrap();
        let (xb, _) = root.selected_submacro(accepted, "xb").unwrap();
        let root_path = MacroHierarchyPath::root("top").unwrap();
        let xa_path = root_path.child("xa").unwrap();
        let xb_path = root_path.child("xb").unwrap();

        assert_eq!(root.accepted().len(), 1);
        assert_eq!(xa.accepted().len(), 1);
        assert_eq!(xb.accepted().len(), 1);
        assert!(!ptr::eq(xa, xb));
        assert!(Arc::ptr_eq(
            result.node(&root_path).unwrap().result().unwrap(),
            result.root_result()
        ));
        assert!(ptr::eq(
            result.node(&xa_path).unwrap().result().unwrap().as_ref(),
            xa
        ));
        assert!(ptr::eq(
            result.node(&xb_path).unwrap().result().unwrap().as_ref(),
            xb
        ));
        assert_eq!(
            result.node(&root_path).unwrap().status(),
            &MacroHierarchyNodeStatus::Refreshed
        );
        assert!(matches!(
            result.node(&xa_path).unwrap().status(),
            MacroHierarchyNodeStatus::ExploredLeaf
        ));
        assert_eq!(result.derivations().count(), 2);
        assert!(
            result
                .node(&root_path)
                .unwrap()
                .preview()
                .unwrap()
                .retained_result()
                .is_none()
        );
        assert_eq!(result.statistics().total_paths(), 3);
        assert_eq!(result.statistics().explored_leaves(), 2);
        assert_eq!(result.statistics().refreshed_parents(), 1);
        assert_eq!(result.statistics().previews_executed(), 1);
        assert_eq!(result.statistics().definitive_evaluations(), 3);

        let selection = result.selection(0).unwrap();
        assert_eq!(selection.root_accepted_index(), 0);
        assert_eq!(
            selection
                .nodes()
                .map(|node| node.path().to_string())
                .collect::<Vec<_>>(),
            ["top", "top.xa", "top.xb"]
        );
        assert_eq!(
            selection
                .instances()
                .map(|instance| instance.path().to_string())
                .collect::<Vec<_>>(),
            ["top::xa", "top::xb"]
        );
    }

    #[test]
    fn explores_the_complete_depth_first_chain_and_keeps_recursive_provenance() {
        let middle = Macro::new(
            "middle",
            Vec::new(),
            Circuit::builder()
                .macro_instance("xleaf", "leaf", Vec::<(&str, &str)>::new())
                .build(),
            Circuit::builder().resistor("rc", "a", "b", 1.0).build(),
        )
        .with_compact_seed(MacroCompactSeed::default());
        let catalog =
            MacroCatalog::from_macros([leaf("leaf"), middle, parent(&[("xmiddle", "middle")])])
                .unwrap();

        let result = explore_macro_hierarchy(
            "top",
            &catalog,
            &PrimitiveCatalog::new(),
            &MacroHierarchyExplorationInput::new(),
        )
        .unwrap();
        let root = result.root_result();
        let (middle_result, middle_candidate) = root
            .selected_submacro(&root.accepted()[0], "xmiddle")
            .unwrap();
        let (leaf_result, _) = middle_result
            .selected_submacro(middle_candidate, "xleaf")
            .unwrap();

        assert_eq!(middle_result.macro_name(), "middle");
        assert_eq!(leaf_result.macro_name(), "leaf");
        assert_eq!(
            result
                .selection(0)
                .unwrap()
                .nodes()
                .map(|node| node.path().to_string())
                .collect::<Vec<_>>(),
            ["top", "top.xmiddle", "top.xmiddle.xleaf"]
        );
    }

    #[test]
    fn derived_child_rejection_removes_every_parent_combination() {
        let child = leaf("stage").with_specification(MacroSpecification::new(
            "requirement",
            MacroSpecificationSource::expression("1.0"),
            MacroSpecificationBounds::unbounded(),
        ));
        let top = parent(&[("xstage", "stage")]).with_derivation_rule(MacroDerivationRule::new(
            "xstage",
            "2.0",
            MacroDerivationReduction::Minimum,
            MacroDerivationTarget::specification_minimum("requirement"),
        ));
        let catalog = MacroCatalog::from_macros([child, top]).unwrap();

        let result = explore_macro_hierarchy(
            "top",
            &catalog,
            &PrimitiveCatalog::new(),
            &MacroHierarchyExplorationInput::new(),
        )
        .unwrap();

        assert!(result.root_result().accepted().is_empty());
        assert_eq!(
            result
                .root_result()
                .candidate_sets()
                .instance("xstage")
                .unwrap()
                .candidates()
                .points
                .len(),
            0
        );
    }

    #[test]
    fn contradictory_derivations_prune_without_becoming_an_execution_error() {
        let child = leaf("stage").with_specification(MacroSpecification::new(
            "requirement",
            MacroSpecificationSource::expression("1.0"),
            MacroSpecificationBounds::unbounded(),
        ));
        let top = parent(&[("xstage", "stage")])
            .with_derivation_rule(MacroDerivationRule::new(
                "xstage",
                "10.0",
                MacroDerivationReduction::Minimum,
                MacroDerivationTarget::specification_minimum("requirement"),
            ))
            .with_derivation_rule(MacroDerivationRule::new(
                "xstage",
                "5.0",
                MacroDerivationReduction::Maximum,
                MacroDerivationTarget::specification_maximum("requirement"),
            ));
        let catalog = MacroCatalog::from_macros([child, top]).unwrap();

        let result = explore_macro_hierarchy(
            "top",
            &catalog,
            &PrimitiveCatalog::new(),
            &MacroHierarchyExplorationInput::new(),
        )
        .unwrap();

        assert!(result.root_result().accepted().is_empty());
        assert!(
            result
                .root_result()
                .candidate_sets()
                .instance("xstage")
                .unwrap()
                .candidates()
                .points
                .is_empty()
        );
        let child_path = MacroHierarchyPath::root("top")
            .unwrap()
            .child("xstage")
            .unwrap();
        assert!(matches!(
            result.node(&child_path).unwrap().status(),
            MacroHierarchyNodeStatus::DerivationPruned { .. }
        ));
        let derivation = result.derivation(&child_path).unwrap();
        assert_eq!(derivation.conditions().audit().len(), 2);
        assert!(derivation.conditions().is_pruned());
        assert_eq!(result.statistics().derivation_pruned(), 1);
        assert!(matches!(
            result.selection(0),
            Err(
                super::super::MacroHierarchySelectionError::RootAcceptedIndexOutOfRange {
                    accepted_count: 0,
                    ..
                }
            )
        ));
    }

    #[test]
    fn full_preview_retention_keeps_the_provisional_arc_only_when_requested() {
        let catalog =
            MacroCatalog::from_macros([leaf("stage"), parent(&[("xstage", "stage")])]).unwrap();
        let mut input = MacroHierarchyExplorationInput::new();
        input.set_retention_policy(super::super::MacroHierarchyRetentionPolicy::FullPreviews);

        let result =
            explore_macro_hierarchy("top", &catalog, &PrimitiveCatalog::new(), &input).unwrap();
        let root = result.node(result.root_path()).unwrap();
        let preview = root.preview().unwrap().retained_result().unwrap();

        assert!(!Arc::ptr_eq(preview, result.root_result()));
        assert_eq!(preview.accepted().len(), 1);
    }

    #[test]
    fn preview_rejection_marks_every_descendant_as_not_reached() {
        let middle = Macro::new(
            "middle",
            Vec::new(),
            Circuit::builder()
                .macro_instance("xleaf", "leaf", Vec::<(&str, &str)>::new())
                .build(),
            Circuit::builder().resistor("rc", "a", "b", 1.0).build(),
        )
        .with_compact_seed(MacroCompactSeed::default());
        let top = parent(&[("xmiddle", "middle")]).with_specification(MacroSpecification::new(
            "reject",
            MacroSpecificationSource::expression("1.0"),
            MacroSpecificationBounds::at_least(2.0),
        ));
        let catalog = MacroCatalog::from_macros([leaf("leaf"), middle, top]).unwrap();

        let result = explore_macro_hierarchy(
            "top",
            &catalog,
            &PrimitiveCatalog::new(),
            &MacroHierarchyExplorationInput::new(),
        )
        .unwrap();
        let root = MacroHierarchyPath::root("top").unwrap();
        let middle = root.child("xmiddle").unwrap();
        let leaf = middle.child("xleaf").unwrap();

        assert_eq!(
            result.node(&root).unwrap().status(),
            &MacroHierarchyNodeStatus::PreviewRejected
        );
        assert_eq!(
            result.node(&middle).unwrap().status(),
            &MacroHierarchyNodeStatus::NotReached {
                blocked_by: Some(root.clone())
            }
        );
        assert_eq!(
            result.node(&leaf).unwrap().status(),
            &MacroHierarchyNodeStatus::NotReached {
                blocked_by: Some(root)
            }
        );
        assert_eq!(result.statistics().preview_rejections(), 1);
        assert_eq!(result.statistics().not_reached(), 2);
        assert!(matches!(
            result.selection(0),
            Err(
                super::super::MacroHierarchySelectionError::RootAcceptedIndexOutOfRange {
                    accepted_count: 0,
                    ..
                }
            )
        ));
    }

    #[test]
    fn rejects_unknown_top_before_executing_any_node() {
        let error = explore_macro_hierarchy(
            "missing",
            &MacroCatalog::new(),
            &PrimitiveCatalog::new(),
            &MacroHierarchyExplorationInput::new(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            MacroHierarchyExplorationError::Validation { .. }
        ));
    }
}
