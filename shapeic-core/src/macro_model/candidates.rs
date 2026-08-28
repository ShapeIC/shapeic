use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use shapeic_lut::DeviceLut;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::circuit::BlockRef;
use crate::exploration::candidate::CandidateSet;
use crate::exploration::filter::{
    CandidateFilterError, CandidateFilterReport, retain_candidate_set,
    retain_candidate_set_with_indices,
};
use crate::primitive::build::{PrimitiveBuildError, build_candidate_set_for_primitive};
use crate::primitive::manifest::PrimitiveManifest;

use super::input::{
    CompactMacroCandidateProvenance, CompactMacroInstanceExplorationInput,
    PrimitiveInstanceExplorationInput,
};
use super::{
    CandidateBuildExecution, Macro, MacroExplorationDefinitionError, MacroExplorationInput,
    MacroExplorationInputValidationError, MacroExplorationInstanceKind,
    validate_macro_exploration_definition, validate_macro_exploration_input,
};

/// Candidates and local filtering statistics for one macro circuit instance.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroInstanceCandidateSet {
    pub(super) instance_path: String,
    pub(super) kind: MacroExplorationInstanceKind,
    pub(super) candidates: CandidateSet,
    pub(super) filter_report: CandidateFilterReport,
    pub(super) compact_provenance: Option<CompactMacroCandidateProvenance>,
}

impl MacroInstanceCandidateSet {
    /// Returns the local instance path used by the macro implementation.
    pub fn instance_path(&self) -> &str {
        &self.instance_path
    }

    /// Returns whether these candidates came from a primitive or compact macro.
    pub const fn kind(&self) -> MacroExplorationInstanceKind {
        self.kind
    }

    /// Returns the filtered candidates used by later combination stages.
    pub const fn candidates(&self) -> &CandidateSet {
        &self.candidates
    }

    /// Returns the local filtering counts for this instance.
    pub const fn filter_report(&self) -> CandidateFilterReport {
        self.filter_report
    }

    /// Returns projected public interface ports for a compact submacro.
    pub fn interface_ports(&self) -> &[String] {
        self.compact_provenance
            .as_ref()
            .map_or(&[], |provenance| provenance.interface_ports.as_slice())
    }
}

/// Candidate sets built for all explorable instances in circuit declaration order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MacroCandidateSets {
    pub(super) instances: Vec<MacroInstanceCandidateSet>,
}

/// Candidate-build timing and row counts for one explorable circuit instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroCandidateBuildInstanceReport {
    instance_path: String,
    duration: Duration,
    input_candidates: usize,
    retained_candidates: usize,
}

impl MacroCandidateBuildInstanceReport {
    pub fn instance_path(&self) -> &str {
        &self.instance_path
    }

    pub const fn duration(&self) -> Duration {
        self.duration
    }

    pub const fn input_candidates(&self) -> usize {
        self.input_candidates
    }

    pub const fn retained_candidates(&self) -> usize {
        self.retained_candidates
    }
}

/// Execution policy and timings produced while constructing macro candidates.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MacroCandidateBuildReport {
    execution: CandidateBuildExecution,
    duration: Duration,
    instances: Vec<MacroCandidateBuildInstanceReport>,
}

impl MacroCandidateBuildReport {
    pub const fn execution(&self) -> CandidateBuildExecution {
        self.execution
    }

    pub const fn duration(&self) -> Duration {
        self.duration
    }

    pub fn instances(&self) -> &[MacroCandidateBuildInstanceReport] {
        &self.instances
    }
}

impl MacroCandidateSets {
    /// Returns instance candidate sets in implementation-circuit order.
    pub fn instances(&self) -> &[MacroInstanceCandidateSet] {
        &self.instances
    }

    /// Finds the candidate set and filtering report for one instance path.
    pub fn instance(&self, instance_path: &str) -> Option<&MacroInstanceCandidateSet> {
        self.instances
            .iter()
            .find(|instance| instance.instance_path == instance_path)
    }

    /// Iterates over candidate sets in the order expected by combination selections.
    pub fn candidate_sets(&self) -> impl ExactSizeIterator<Item = &CandidateSet> {
        self.instances
            .iter()
            .map(MacroInstanceCandidateSet::candidates)
    }

    /// Returns the number of explorable instances represented by this result.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Returns whether the macro has no primitive or compact-macro candidates.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

/// Errors produced while constructing and filtering macro candidate sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroCandidateBuildError {
    InvalidDefinition {
        errors: Vec<MacroExplorationDefinitionError>,
    },
    InvalidInput {
        errors: Vec<MacroExplorationInputValidationError>,
    },
    PrimitiveBuild {
        macro_name: String,
        instance_path: String,
        primitive: String,
        error: PrimitiveBuildError,
    },
    CandidateFilter {
        macro_name: String,
        instance_path: String,
        error: CandidateFilterError,
    },
    CandidateThreadPool {
        reason: String,
    },
}

impl fmt::Display for MacroCandidateBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDefinition { errors } => {
                write!(
                    formatter,
                    "macro exploration definition has {} validation error(s)",
                    errors.len()
                )?;
                for error in errors {
                    write!(formatter, "; {error}")?;
                }
                Ok(())
            }
            Self::InvalidInput { errors } => {
                write!(
                    formatter,
                    "macro exploration input has {} validation error(s)",
                    errors.len()
                )?;
                for error in errors {
                    write!(formatter, "; {error}")?;
                }
                Ok(())
            }
            Self::PrimitiveBuild {
                macro_name,
                instance_path,
                primitive,
                error,
            } => write!(
                formatter,
                "could not build macro '{macro_name}' instance '{instance_path}' primitive '{primitive}': {error:?}"
            ),
            Self::CandidateFilter {
                macro_name,
                instance_path,
                error,
            } => write!(
                formatter,
                "could not filter macro '{macro_name}' instance '{instance_path}': {error}"
            ),
            Self::CandidateThreadPool { reason } => {
                write!(formatter, "could not build candidate worker pool: {reason}")
            }
        }
    }
}

impl Error for MacroCandidateBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CandidateFilter { error, .. } => Some(error),
            Self::InvalidDefinition { .. }
            | Self::InvalidInput { .. }
            | Self::PrimitiveBuild { .. }
            | Self::CandidateThreadPool { .. } => None,
        }
    }
}

/// Builds and locally filters every primitive and compact-submacro candidate set.
///
/// The input is consumed so precomputed submacro candidate sets can be adopted
/// and filtered in place without cloning their rows. Linear circuit elements do
/// not produce candidate dimensions and are omitted from the returned order.
pub fn build_macro_candidate_sets(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    input: MacroExplorationInput<'_>,
) -> Result<MacroCandidateSets, MacroCandidateBuildError> {
    build_macro_candidate_sets_with_execution(
        macro_,
        primitive_catalog,
        input,
        CandidateBuildExecution::Sequential,
    )
    .map(|(candidates, _)| candidates)
}

/// Builds candidate sets using an explicit per-instance execution policy.
pub fn build_macro_candidate_sets_with_execution(
    macro_: &Macro,
    primitive_catalog: &PrimitiveCatalog,
    mut input: MacroExplorationInput<'_>,
    execution: CandidateBuildExecution,
) -> Result<(MacroCandidateSets, MacroCandidateBuildReport), MacroCandidateBuildError> {
    let build_start = Instant::now();
    let definition_errors = validate_macro_exploration_definition(macro_, primitive_catalog);
    if !definition_errors.is_empty() {
        return Err(MacroCandidateBuildError::InvalidDefinition {
            errors: definition_errors,
        });
    }
    let validation_errors = validate_macro_exploration_input(macro_, primitive_catalog, &input);
    if !validation_errors.is_empty() {
        return Err(MacroCandidateBuildError::InvalidInput {
            errors: validation_errors,
        });
    }
    input.resolve_primitive_defaults(macro_);

    let mut tasks = Vec::new();
    for (circuit_index, instance) in macro_.circuit().instances().iter().enumerate() {
        let task = match instance.block() {
            BlockRef::Primitive(primitive_name) => {
                let primitive = primitive_catalog
                    .get(primitive_name)
                    .expect("macro exploration input validation resolved the primitive");
                let device_type = primitive
                    .transistor_type
                    .as_deref()
                    .expect("macro exploration input validation resolved the device type");
                let model = input
                    .device_models
                    .get(device_type)
                    .copied()
                    .expect("macro exploration input validation resolved the device model");
                let primitive_input = input
                    .primitive_instances
                    .remove(instance.name())
                    .expect("macro exploration input validation resolved the primitive input");
                CandidateBuildTask::Primitive {
                    circuit_index,
                    instance_path: instance.name().to_owned(),
                    primitive_name: primitive_name.clone(),
                    model,
                    primitive: primitive.clone(),
                    input: primitive_input,
                }
            }
            BlockRef::Macro(_) => {
                let compact_input = input
                    .compact_macro_instances
                    .remove(instance.name())
                    .expect("macro exploration input validation resolved the compact input");
                CandidateBuildTask::CompactMacro {
                    circuit_index,
                    instance_path: instance.name().to_owned(),
                    input: compact_input,
                }
            }
            BlockRef::Element(_) => continue,
        };
        tasks.push(task);
    }

    let macro_name = macro_.name();
    let results = match execution {
        CandidateBuildExecution::Sequential => tasks
            .into_iter()
            .map(|task| build_candidate_task(macro_name, task))
            .collect::<Vec<_>>(),
        CandidateBuildExecution::Parallel { workers } => {
            let pool = ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .map_err(|error| MacroCandidateBuildError::CandidateThreadPool {
                    reason: error.to_string(),
                })?;
            pool.install(|| {
                tasks
                    .into_par_iter()
                    .map(|task| build_candidate_task(macro_name, task))
                    .collect::<Vec<_>>()
            })
        }
    };
    let mut built = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    built.sort_by_key(|result| result.circuit_index);
    let (instances, instance_reports) = built
        .into_iter()
        .map(|result| (result.candidates, result.report))
        .unzip();
    let report = MacroCandidateBuildReport {
        execution,
        duration: build_start.elapsed(),
        instances: instance_reports,
    };
    Ok((MacroCandidateSets { instances }, report))
}

enum CandidateBuildTask<'lut> {
    Primitive {
        circuit_index: usize,
        instance_path: String,
        primitive_name: String,
        model: &'lut DeviceLut,
        primitive: PrimitiveManifest,
        input: PrimitiveInstanceExplorationInput,
    },
    CompactMacro {
        circuit_index: usize,
        instance_path: String,
        input: CompactMacroInstanceExplorationInput,
    },
}

struct BuiltCandidateTask {
    circuit_index: usize,
    candidates: MacroInstanceCandidateSet,
    report: MacroCandidateBuildInstanceReport,
}

fn build_candidate_task(
    macro_name: &str,
    task: CandidateBuildTask<'_>,
) -> Result<BuiltCandidateTask, MacroCandidateBuildError> {
    let start = Instant::now();
    let (circuit_index, instance_path, kind, mut candidates, filters, mut provenance) = match task {
        CandidateBuildTask::Primitive {
            circuit_index,
            instance_path,
            primitive_name,
            model,
            primitive,
            input,
        } => {
            let candidates = build_candidate_set_for_primitive(
                model,
                &primitive,
                &instance_path,
                input.build_input,
            )
            .map_err(|error| MacroCandidateBuildError::PrimitiveBuild {
                macro_name: macro_name.to_owned(),
                instance_path: instance_path.clone(),
                primitive: primitive_name,
                error,
            })?;
            (
                circuit_index,
                instance_path,
                MacroExplorationInstanceKind::Primitive,
                candidates,
                input.filters,
                None,
            )
        }
        CandidateBuildTask::CompactMacro {
            circuit_index,
            instance_path,
            input,
        } => (
            circuit_index,
            instance_path,
            MacroExplorationInstanceKind::CompactMacro,
            input.candidates,
            input.filters,
            input.provenance,
        ),
    };
    let input_candidates = candidates.points.len();
    let filter_error = |error| MacroCandidateBuildError::CandidateFilter {
        macro_name: macro_name.to_owned(),
        instance_path: instance_path.clone(),
        error,
    };
    let filter_report = if let Some(provenance) = &mut provenance {
        debug_assert_eq!(provenance.accepted_indices.len(), candidates.points.len());
        let (report, retained_indices) =
            retain_candidate_set_with_indices(&mut candidates, &filters).map_err(filter_error)?;
        provenance.accepted_indices = retained_indices
            .into_iter()
            .map(|index| provenance.accepted_indices[index])
            .collect();
        report
    } else {
        retain_candidate_set(&mut candidates, &filters).map_err(filter_error)?
    };
    let retained_candidates = candidates.points.len();
    Ok(BuiltCandidateTask {
        circuit_index,
        candidates: MacroInstanceCandidateSet {
            instance_path: instance_path.clone(),
            kind,
            candidates,
            filter_report,
            compact_provenance: provenance,
        },
        report: MacroCandidateBuildInstanceReport {
            instance_path,
            duration: start.elapsed(),
            input_candidates,
            retained_candidates,
        },
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use shapeic_lut::LookupTable;

    use crate::circuit::Circuit;
    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::exploration::filter::CandidateFilter;
    use crate::primitive::build::{
        BuildExpression, PrimitiveBuildInput, PrimitiveBuildInputKind, PrimitiveBuildInputSpec,
        PrimitiveBuildSpec, PrimitiveBuildValue, SweepMode,
    };
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};

    use super::*;
    use crate::macro_model::{
        CompactMacroInstanceExplorationInput, MacroDesignVariable, MacroDesignVariableBinding,
        MacroPrimitiveDefault, PrimitiveInstanceExplorationInput,
    };

    fn primitive() -> PrimitiveManifest {
        PrimitiveManifest {
            name: "stage".to_owned(),
            version: "1.0".to_owned(),
            description: None,
            subckt_name: "stage".to_owned(),
            pins: vec![Pin {
                name: "OUT".to_owned(),
                role: PinRole::Output,
            }],
            files: PrimitiveFiles {
                netlist: "netlist.spice".to_owned(),
                build: None,
                symbol: None,
            },
            small_signal: None,
            physical_model: None,
            transistor_type: Some("nmos".to_owned()),
            layout_params: None,
            lut_config: None,
            build: Some(PrimitiveBuildSpec {
                inputs: vec![PrimitiveBuildInputSpec {
                    name: "width".to_owned(),
                    kind: PrimitiveBuildInputKind::Vector,
                    required: true,
                    source: None,
                }],
                sweep_mode: SweepMode::Aligned,
                derived: Vec::new(),
                lut: Vec::new(),
                columns: vec![BuildExpression {
                    name: "width".to_owned(),
                    expr: "width".to_owned(),
                }],
            }),
        }
    }

    fn macro_() -> Macro {
        Macro::new(
            "top",
            Vec::new(),
            Circuit::builder()
                .primitive("xstage", "stage", [("OUT", "VOUT")])
                .resistor("rout", "VOUT", "0", 1.0e6)
                .macro_instance("xload", "load", [("OUT", "VOUT")])
                .build(),
            Circuit::builder()
                .resistor("rout", "VOUT", "0", 1.0e6)
                .build(),
        )
    }

    fn fixture() -> LookupTable {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../shapeic-lut/tests/fixtures/sstadex_combined.npz");
        LookupTable::open(path).expect("LUT fixture should load")
    }

    fn input<'lut>(model: &'lut shapeic_lut::DeviceLut) -> MacroExplorationInput<'lut> {
        let mut input = MacroExplorationInput::new();
        input.register_device_model("nmos", model).unwrap();
        input
            .register_primitive_instance(
                "xstage",
                PrimitiveInstanceExplorationInput::new(
                    PrimitiveBuildInput::new(HashMap::from([(
                        "width".to_owned(),
                        PrimitiveBuildValue::Vector(vec![1.0, 2.0, 3.0]),
                    )])),
                    vec![CandidateFilter::at_most("xstage.width", 2.0).unwrap()],
                ),
            )
            .unwrap();
        input
            .register_compact_macro_instance(
                "xload",
                CompactMacroInstanceExplorationInput::new(
                    CandidateSet::new(
                        "load",
                        vec![
                            CandidatePoint::new(vec![("xload.score".to_owned(), 1.0)]),
                            CandidatePoint::new(vec![("xload.score".to_owned(), 3.0)]),
                        ],
                    ),
                    vec![CandidateFilter::at_least("xload.score", 2.0).unwrap()],
                ),
            )
            .unwrap();
        input
    }

    #[test]
    fn builds_and_filters_instances_in_circuit_order_while_skipping_elements() {
        let table = fixture();
        let model = table.model("fixture_nmos").unwrap();
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(primitive());

        let candidates = build_macro_candidate_sets(&macro_(), &catalog, input(model)).unwrap();

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates.instances()[0].instance_path(), "xstage");
        assert_eq!(candidates.instances()[1].instance_path(), "xload");
        let stage = candidates.instance("xstage").unwrap();
        assert_eq!(stage.kind(), MacroExplorationInstanceKind::Primitive);
        assert_eq!(stage.filter_report().input_count(), 3);
        assert_eq!(stage.filter_report().retained_count(), 2);
        assert_eq!(stage.candidates().points[1].get("xstage.width"), Some(2.0));
        let load = candidates.instance("xload").unwrap();
        assert_eq!(load.kind(), MacroExplorationInstanceKind::CompactMacro);
        assert_eq!(load.filter_report().input_count(), 2);
        assert_eq!(load.filter_report().retained_count(), 1);
        assert_eq!(load.candidates().points[0].get("xload.score"), Some(3.0));
    }

    #[test]
    fn builds_primitive_candidates_from_macro_defaults_and_public_overrides() {
        let table = fixture();
        let model = table.model("fixture_nmos").unwrap();
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(primitive());
        let macro_ = macro_()
            .with_primitive_default(MacroPrimitiveDefault::new(
                "xstage",
                PrimitiveBuildInput::new(HashMap::from([(
                    "width".to_owned(),
                    PrimitiveBuildValue::Vector(vec![1.0, 2.0, 3.0]),
                )])),
                vec![CandidateFilter::at_most("xstage.width", 3.0).unwrap()],
            ))
            .with_design_variable(MacroDesignVariable::new(
                "stage_widths",
                PrimitiveBuildInputKind::Vector,
                vec![MacroDesignVariableBinding::new("xstage", "width")],
            ));
        let mut input = MacroExplorationInput::new();
        input.register_device_model("nmos", model).unwrap();
        input
            .register_design_variable_override(
                "stage_widths",
                PrimitiveBuildValue::Vector(vec![2.0, 3.0, 4.0]),
            )
            .unwrap();
        input
            .register_compact_macro_instance(
                "xload",
                CompactMacroInstanceExplorationInput::new(
                    CandidateSet::new(
                        "load",
                        vec![CandidatePoint::new(vec![("xload.score".to_owned(), 1.0)])],
                    ),
                    Vec::new(),
                ),
            )
            .unwrap();

        let candidates = build_macro_candidate_sets(&macro_, &catalog, input).unwrap();

        let stage = candidates.instance("xstage").unwrap();
        assert_eq!(stage.filter_report().input_count(), 3);
        assert_eq!(stage.filter_report().retained_count(), 2);
        assert_eq!(stage.candidates().points[0].get("xstage.width"), Some(2.0));
        assert_eq!(stage.candidates().points[1].get("xstage.width"), Some(3.0));
    }

    #[test]
    fn parallel_build_matches_sequential_order_candidates_and_counts() {
        let table = fixture();
        let model = table.model("fixture_nmos").unwrap();
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(primitive());

        let (sequential, sequential_report) = build_macro_candidate_sets_with_execution(
            &macro_(),
            &catalog,
            input(model),
            CandidateBuildExecution::Sequential,
        )
        .unwrap();
        let (parallel, parallel_report) = build_macro_candidate_sets_with_execution(
            &macro_(),
            &catalog,
            input(model),
            CandidateBuildExecution::Parallel { workers: 2 },
        )
        .unwrap();

        assert_eq!(parallel, sequential);
        assert_eq!(parallel_report.instances().len(), 2);
        for (parallel, sequential) in parallel_report
            .instances()
            .iter()
            .zip(sequential_report.instances())
        {
            assert_eq!(parallel.instance_path(), sequential.instance_path());
            assert_eq!(parallel.input_candidates(), sequential.input_candidates());
            assert_eq!(
                parallel.retained_candidates(),
                sequential.retained_candidates()
            );
        }
    }

    #[test]
    fn parallel_build_reports_the_first_error_in_circuit_order() {
        let table = fixture();
        let model = table.model("fixture_nmos").unwrap();
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(primitive());
        let mut input = input(model);
        input.primitive_instances.get_mut("xstage").unwrap().filters =
            vec![CandidateFilter::at_most("xstage.first_missing", 2.0).unwrap()];
        input
            .compact_macro_instances
            .get_mut("xload")
            .unwrap()
            .filters = vec![CandidateFilter::at_most("xload.second_missing", 2.0).unwrap()];

        let error = build_macro_candidate_sets_with_execution(
            &macro_(),
            &catalog,
            input,
            CandidateBuildExecution::Parallel { workers: 2 },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            MacroCandidateBuildError::CandidateFilter { instance_path, .. }
                if instance_path == "xstage"
        ));
    }

    #[test]
    fn adds_macro_and_instance_context_to_filter_errors() {
        let table = fixture();
        let model = table.model("fixture_nmos").unwrap();
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(primitive());
        let mut input = input(model);
        input.primitive_instances.get_mut("xstage").unwrap().filters =
            vec![CandidateFilter::at_most("xstage.missing", 2.0).unwrap()];

        let error = build_macro_candidate_sets(&macro_(), &catalog, input).unwrap_err();

        assert!(matches!(
            error,
            MacroCandidateBuildError::CandidateFilter {
                macro_name,
                instance_path,
                error: CandidateFilterError::MissingColumn { column, .. },
            } if macro_name == "top"
                && instance_path == "xstage"
                && column == "xstage.missing"
        ));
    }

    #[test]
    fn adds_macro_and_instance_context_to_primitive_build_errors() {
        let table = fixture();
        let model = table.model("fixture_nmos").unwrap();
        let mut primitive = primitive();
        primitive.build = None;
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(primitive);

        let error = build_macro_candidate_sets(&macro_(), &catalog, input(model)).unwrap_err();

        assert!(matches!(
            error,
            MacroCandidateBuildError::PrimitiveBuild {
                macro_name,
                instance_path,
                primitive,
                error: PrimitiveBuildError::MissingBuildSpec { primitive: missing },
            } if macro_name == "top"
                && instance_path == "xstage"
                && primitive == "stage"
                && missing == "stage"
        ));
    }
}
