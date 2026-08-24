//! Preparation of macro-owned AC testbenches.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::PathBuf;

use shapeic_mna::numeric::NumericMnaError;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::testbench::{AcAnalysis, AcTestbench, AcTestbenchBuildError, PreparedAcTestbench};

use super::{
    Macro, MacroAcTestbench, MacroAnalysisDomain, MacroCatalog, MacroRenderError, MacroRenderMode,
    MacroTestbenchSource, ResolvedPhysicalPrimitive, ResolvedPrimitiveBranch,
    render_small_signal_netlist,
};

type ComposedMacroAcTestbench = (
    String,
    Vec<String>,
    Vec<ResolvedPrimitiveBranch>,
    Vec<ResolvedPhysicalPrimitive>,
);

/// Compiled macro AC testbench and its resolved expanded primitive topology.
#[derive(Clone, Debug)]
pub struct PreparedMacroAcTestbench {
    testbench: PreparedAcTestbench,
    domain: MacroAnalysisDomain,
    primitive_branches: Vec<ResolvedPrimitiveBranch>,
    physical_primitives: Vec<ResolvedPhysicalPrimitive>,
}

impl PreparedMacroAcTestbench {
    /// Returns the numerical parameter names in candidate binding order.
    pub fn parameter_names(&self) -> &[String] {
        self.testbench.parameter_names()
    }

    /// Returns the AC analysis attached to the macro testbench.
    pub const fn analysis(&self) -> &AcAnalysis {
        self.testbench.analysis()
    }

    /// Returns the physical modeling domain selected for this testbench.
    pub const fn domain(&self) -> MacroAnalysisDomain {
        self.domain
    }

    /// Returns the primitive branches materialized by the selected render mode.
    pub fn primitive_branches(&self) -> &[ResolvedPrimitiveBranch] {
        &self.primitive_branches
    }

    /// Returns resolved physical primitive query and stamp plans.
    pub fn physical_primitives(&self) -> &[ResolvedPhysicalPrimitive] {
        &self.physical_primitives
    }

    /// Instantiates the compiled MNA for one candidate parameter vector.
    pub fn instantiate(
        &mut self,
        parameter_values: &[f64],
    ) -> Result<AcTestbench, NumericMnaError> {
        self.testbench.instantiate(parameter_values)
    }
}

/// Errors produced while preparing a macro-owned AC testbench.
#[derive(Debug)]
pub enum MacroTestbenchPrepareError {
    /// The selected macro representation could not be rendered.
    Render(MacroRenderError),
    /// A file-backed testbench could not be read.
    ReadTestbench {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The composed linear netlist could not be compiled for numerical MNA.
    Build(AcTestbenchBuildError),
}

impl fmt::Display for MacroTestbenchPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Render(error) => write!(formatter, "could not render macro testbench: {error}"),
            Self::ReadTestbench { path, source } => write!(
                formatter,
                "could not read macro testbench '{}': {source}",
                path.display()
            ),
            Self::Build(error) => write!(formatter, "could not prepare macro testbench: {error}"),
        }
    }
}

impl Error for MacroTestbenchPrepareError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Render(error) => Some(error),
            Self::ReadTestbench { source, .. } => Some(source),
            Self::Build(error) => Some(error),
        }
    }
}

/// Renders, composes, and compiles one macro-owned AC testbench.
///
/// The symbolic MNA and its numerical evaluator are prepared once. The
/// returned testbench can then be instantiated repeatedly using values in its
/// reported [`PreparedMacroAcTestbench::parameter_names`] order. Expanded
/// primitive topology is retained for candidate-dependent numerical stamps.
pub fn prepare_macro_ac_testbench(
    macro_: &Macro,
    testbench: &MacroAcTestbench,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
    render_mode: MacroRenderMode,
) -> Result<PreparedMacroAcTestbench, MacroTestbenchPrepareError> {
    let domain = testbench.domain();
    let (source, parameter_order, primitive_branches, physical_primitives) =
        compose_macro_ac_testbench(
            macro_,
            testbench,
            primitive_catalog,
            macro_catalog,
            render_mode,
        )?;
    let parameter_order = parameter_order
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();

    let testbench =
        PreparedAcTestbench::from_spice(&source, &parameter_order, testbench.analysis().clone())
            .map_err(MacroTestbenchPrepareError::Build)?;
    Ok(PreparedMacroAcTestbench {
        testbench,
        domain,
        primitive_branches,
        physical_primitives,
    })
}

fn compose_macro_ac_testbench(
    macro_: &Macro,
    testbench: &MacroAcTestbench,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
    render_mode: MacroRenderMode,
) -> Result<ComposedMacroAcTestbench, MacroTestbenchPrepareError> {
    let rendered =
        render_small_signal_netlist(macro_, primitive_catalog, macro_catalog, render_mode)
            .map_err(MacroTestbenchPrepareError::Render)?;
    let testbench_source = read_testbench_source(testbench.source())?;
    let (macro_source, parameter_order, primitive_branches, physical_primitives) =
        rendered.into_parts();
    let source = compose_sources(macro_source, testbench.name(), &testbench_source);
    Ok((
        source,
        parameter_order,
        primitive_branches,
        physical_primitives,
    ))
}

fn read_testbench_source(
    source: &MacroTestbenchSource,
) -> Result<Cow<'_, str>, MacroTestbenchPrepareError> {
    match source {
        MacroTestbenchSource::Spice(source) => Ok(Cow::Borrowed(source)),
        MacroTestbenchSource::SpiceFile(path) => {
            fs::read_to_string(path).map(Cow::Owned).map_err(|source| {
                MacroTestbenchPrepareError::ReadTestbench {
                    path: path.clone(),
                    source,
                }
            })
        }
    }
}

fn compose_sources(mut source: String, testbench_name: &str, testbench_source: &str) -> String {
    source.truncate(source.trim_end().len());
    source.reserve(testbench_source.len() + testbench_name.len() + 32);
    if !source.is_empty() {
        source.push('\n');
    }
    source.push_str("* testbench ");
    source.push_str(testbench_name);
    source.push('\n');
    source.push_str(testbench_source.trim());
    source.push('\n');
    source
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use crate::analysis::{
        AcMetric, AcMetricSet, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets,
    };
    use crate::catalog::primitive_catalog::PrimitiveCatalog;
    use crate::circuit::Circuit;
    use crate::exploration::candidate::{CandidatePoint, CandidateSet};
    use crate::macro_model::{
        Macro, MacroAcTestbench, MacroAnalysisDomain, MacroCatalog, MacroCompactOutputBinding,
        MacroOutputSource, MacroPort, MacroPortRole, MacroRenderMode, MacroTestbenchPrepareError,
        PreparedMacroAcCandidateEvaluator,
    };
    use crate::netlist::names::small_signal_param_name;
    use crate::primitive::build::{LutBuildSpec, LutLengths, PrimitiveBuildSpec, SweepMode};
    use crate::primitive::manifest::{
        Pin, PinRole, PrimitiveFiles, PrimitiveManifest, PrimitivePhysicalModel,
    };
    use crate::primitive::small_signal::{SmallSignalBranch, SmallSignalModel};
    use crate::testbench::{AcAnalysis, TransferFunction};
    use shapeic_lut::{MosCapacitanceMatrix, MosExtrinsicCapacitances};

    use super::{compose_macro_ac_testbench, compose_sources, prepare_macro_ac_testbench};

    fn analysis() -> AcAnalysis {
        AcAnalysis::new(
            TransferFunction::new("VIN", "VOUT"),
            AdaptiveAcConfig {
                min_frequency_hz: 1.0,
                max_frequency_hz: 1.0e6,
                coarse_points_per_decade: 4,
                crossing_relative_tolerance: 1.0e-4,
                max_refinement_steps: 16,
                retain_samples: false,
            },
            AdaptiveAcPolicy {
                mode: AnalysisMode::FullInsight,
                targets: AnalysisTargets::NONE,
                metrics: AcMetricSet::from_metric(AcMetric::DcGainDb),
            },
        )
    }

    fn primitive_catalog() -> PrimitiveCatalog {
        let mut catalog = PrimitiveCatalog::new();
        catalog.register(PrimitiveManifest {
            name: "gain_primitive".to_owned(),
            version: "1.0".to_owned(),
            description: None,
            subckt_name: "gain_primitive".to_owned(),
            pins: vec![
                Pin {
                    name: "VIN".to_owned(),
                    role: PinRole::Input,
                },
                Pin {
                    name: "VOUT".to_owned(),
                    role: PinRole::Output,
                },
                Pin {
                    name: "VSS".to_owned(),
                    role: PinRole::Supply,
                },
            ],
            files: PrimitiveFiles {
                netlist: "gain.spice".to_owned(),
                build: None,
                symbol: None,
            },
            small_signal: Some(SmallSignalModel::new(vec![SmallSignalBranch::new(
                "m1", "VOUT", "VIN", "VSS", "VSS",
            )])),
            physical_model: Some(PrimitivePhysicalModel::new(
                "gain_physical",
                "m1",
                [("IN", "VIN"), ("OUT", "VOUT"), ("GND", "VSS")],
            )),
            transistor_type: None,
            layout_params: None,
            lut_config: None,
            build: Some(PrimitiveBuildSpec {
                inputs: Vec::new(),
                sweep_mode: SweepMode::Aligned,
                derived: Vec::new(),
                lut: vec![LutBuildSpec {
                    name: "m1".to_owned(),
                    device: "nmos".to_owned(),
                    current: "current".to_owned(),
                    dof: HashMap::new(),
                    lengths: Some(LutLengths::Values(vec![1.0])),
                }],
                columns: Vec::new(),
            }),
        });
        catalog
    }

    fn macro_() -> Macro {
        Macro::new(
            "gain_stage",
            vec![
                MacroPort::new("VIN", MacroPortRole::Input),
                MacroPort::new("VOUT", MacroPortRole::Output),
                MacroPort::new("VSS", MacroPortRole::Ground),
            ],
            Circuit::builder()
                .primitive(
                    "xcore",
                    "gain_primitive",
                    [("VIN", "VIN"), ("VOUT", "VOUT"), ("VSS", "VSS")],
                )
                .resistor("rload", "VOUT", "VSS", "load_resistance")
                .build(),
            Circuit::builder()
                .vccs("gm", "VOUT", "VSS", "VIN", "VSS", "gm_eq")
                .build(),
        )
        .with_compact_output(MacroCompactOutputBinding::new(
            "gm_eq",
            MacroOutputSource::candidate_column(
                "xcore",
                small_signal_param_name("gm", "xcore", "m1"),
            ),
        ))
    }

    fn candidate_set() -> CandidateSet {
        let mut values = vec![
            ("gm__xcore__m1".to_owned(), 1.0e-3),
            ("ro__xcore__m1".to_owned(), 1.0e5),
            ("load_resistance".to_owned(), 1.0e4),
        ];
        values.extend(
            MosCapacitanceMatrix::INDEPENDENT_PARAMETERS
                .into_iter()
                .chain(MosExtrinsicCapacitances::PARAMETERS)
                .enumerate()
                .map(|(index, parameter)| {
                    (
                        small_signal_param_name(parameter, "xcore", "m1"),
                        (index + 1) as f64 * 1.0e-15,
                    )
                }),
        );
        values.extend(
            [
                ("length", 1.5),
                ("finger_width", 1.5),
                ("nf", 1.0),
                ("vbs", -0.5),
                ("vgs", 1.0),
                ("vds", 2.0),
            ]
            .map(|(parameter, value)| {
                (small_signal_param_name(parameter, "xcore", "m1"), value)
            }),
        );
        CandidateSet::new("xcore", vec![CandidatePoint::new(values)])
    }

    #[test]
    fn composes_macro_and_testbench_sources_with_a_stable_boundary() {
        assert_eq!(
            compose_sources("R1 OUT VSS r\n".to_owned(), "gain", "V1 IN VSS 1\n.end\n"),
            "R1 OUT VSS r\n* testbench gain\nV1 IN VSS 1\n.end\n"
        );
    }

    #[test]
    fn composes_inline_testbench_with_the_rendered_parameter_order() {
        let macro_ = macro_();
        let catalog = MacroCatalog::from_macros([macro_.clone()]).unwrap();
        let testbench = MacroAcTestbench::from_spice("gain", "Vinput VIN VSS 1\n", analysis());

        let (source, parameter_order, primitive_branches, physical_primitives) =
            compose_macro_ac_testbench(
                &macro_,
                &testbench,
                &primitive_catalog(),
                &catalog,
                MacroRenderMode::Expanded,
            )
            .unwrap();

        assert_eq!(
            parameter_order,
            ["gm__xcore__m1", "ro__xcore__m1", "load_resistance"]
        );
        assert!(source.contains("G_gm__xcore__m1 VOUT VSS VIN VSS gm__xcore__m1"));
        assert!(source.contains("* testbench gain\nVinput VIN VSS 1\n"));
        let [branch] = primitive_branches.as_slice() else {
            panic!("expected one resolved primitive branch");
        };
        assert_eq!(branch.instance_path(), "xcore");
        assert_eq!(branch.branch_name(), "m1");
        assert_eq!(branch.gate_node(), "VIN");
        assert_eq!(branch.drain_node(), "VOUT");
        assert_eq!(branch.source_node(), "VSS");
        assert_eq!(branch.bulk_node(), "VSS");
        let [physical] = physical_primitives.as_slice() else {
            panic!("expected one compiled physical primitive plan");
        };
        assert_eq!(physical.instance_path(), "xcore");
        assert_eq!(physical.lut_primitive(), "gain_physical");
        assert_eq!(physical.ports()[1].node(), "VOUT");
    }

    #[test]
    fn reports_the_path_when_a_file_backed_testbench_cannot_be_read() {
        let macro_ = macro_();
        let catalog = MacroCatalog::from_macros([macro_.clone()]).unwrap();
        let missing = PathBuf::from("missing/macro_testbench.spice");
        let testbench = MacroAcTestbench::from_spice_file("gain", &missing, analysis());

        let error = compose_macro_ac_testbench(
            &macro_,
            &testbench,
            &primitive_catalog(),
            &catalog,
            MacroRenderMode::Expanded,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            MacroTestbenchPrepareError::ReadTestbench { path, .. } if path == missing
        ));
    }

    #[test]
    fn instantiates_and_analyzes_a_candidate_with_resolved_capacitances() {
        let macro_ = macro_();
        let catalog = MacroCatalog::from_macros([macro_.clone()]).unwrap();
        let testbench = MacroAcTestbench::from_spice(
            "gain",
            "Vinput VIN VSS 1\n.end\n",
            analysis(),
        )
        .with_domain(MacroAnalysisDomain::LayoutAware);
        let prepared = prepare_macro_ac_testbench(
            &macro_,
            &testbench,
            &primitive_catalog(),
            &catalog,
            MacroRenderMode::Expanded,
        )
        .unwrap();
        assert_eq!(prepared.physical_primitives().len(), 1);
        assert_eq!(prepared.domain(), MacroAnalysisDomain::LayoutAware);
        let layout_prepared = prepared.clone();
        let mut electrical_prepared = prepared;
        electrical_prepared.domain = MacroAnalysisDomain::Electrical;
        let candidates = candidate_set();
        let mut evaluator = PreparedMacroAcCandidateEvaluator::new(
            electrical_prepared,
            &[&candidates],
        )
        .unwrap();

        let electrical_candidate = evaluator.instantiate(&[0]).unwrap();

        assert!(
            electrical_candidate
                .system()
                .capacitance_matrix()
                .iter()
                .any(|value| *value != 0.0)
        );

        let missing_lut_error = PreparedMacroAcCandidateEvaluator::new(
            layout_prepared.clone(),
            &[&candidates],
        )
        .unwrap_err();
        assert!(matches!(
            missing_lut_error,
            crate::macro_model::PreparedMacroAcCandidateEvaluatorError::MissingPhysicalLut
        ));
        let physical_lut = crate::macro_model::physical::tests::physical_lut_for(
            "gain_physical",
            &["IN", "OUT", "GND"],
        );
        let mut layout_evaluator = PreparedMacroAcCandidateEvaluator::new_with_physical_lut(
            layout_prepared,
            &[&candidates],
            Some(&physical_lut),
        )
        .unwrap();
        let layout_candidate = layout_evaluator.instantiate(&[0]).unwrap();
        assert_ne!(
            layout_candidate.system().base_matrix(),
            electrical_candidate.system().base_matrix()
        );
        assert_ne!(
            layout_candidate.system().capacitance_matrix(),
            electrical_candidate.system().capacitance_matrix()
        );

        let outcome = evaluator.analyze(&[0]).unwrap();
        assert!(outcome.metrics.dc_gain_db.is_some());
    }
}
