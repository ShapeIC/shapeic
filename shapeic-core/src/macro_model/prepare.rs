//! Preparation of macro-owned AC testbenches.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::PathBuf;

use crate::catalog::primitive_catalog::PrimitiveCatalog;
use crate::testbench::{AcTestbenchBuildError, PreparedAcTestbench};

use super::{
    Macro, MacroAcTestbench, MacroCatalog, MacroRenderError, MacroRenderMode, MacroTestbenchSource,
    render_small_signal_netlist,
};

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
/// reported [`PreparedAcTestbench::parameter_names`] order.
pub fn prepare_macro_ac_testbench(
    macro_: &Macro,
    testbench: &MacroAcTestbench,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
    render_mode: MacroRenderMode,
) -> Result<PreparedAcTestbench, MacroTestbenchPrepareError> {
    let (source, parameter_order) = compose_macro_ac_testbench(
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

    PreparedAcTestbench::from_spice(&source, &parameter_order, testbench.analysis().clone())
        .map_err(MacroTestbenchPrepareError::Build)
}

fn compose_macro_ac_testbench(
    macro_: &Macro,
    testbench: &MacroAcTestbench,
    primitive_catalog: &PrimitiveCatalog,
    macro_catalog: &MacroCatalog,
    render_mode: MacroRenderMode,
) -> Result<(String, Vec<String>), MacroTestbenchPrepareError> {
    let rendered =
        render_small_signal_netlist(macro_, primitive_catalog, macro_catalog, render_mode)
            .map_err(MacroTestbenchPrepareError::Render)?;
    let testbench_source = read_testbench_source(testbench.source())?;
    let (macro_source, parameter_order) = rendered.into_parts();
    let source = compose_sources(macro_source, testbench.name(), &testbench_source);
    Ok((source, parameter_order))
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
    use std::path::PathBuf;

    use crate::analysis::{
        AcMetric, AcMetricSet, AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets,
    };
    use crate::catalog::primitive_catalog::PrimitiveCatalog;
    use crate::circuit::Circuit;
    use crate::macro_model::{
        Macro, MacroAcTestbench, MacroCatalog, MacroPort, MacroPortRole, MacroRenderMode,
        MacroTestbenchPrepareError,
    };
    use crate::primitive::manifest::{Pin, PinRole, PrimitiveFiles, PrimitiveManifest};
    use crate::primitive::small_signal::{SmallSignalBranch, SmallSignalModel};
    use crate::testbench::{AcAnalysis, TransferFunction};

    use super::{compose_macro_ac_testbench, compose_sources};

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
            transistor_type: None,
            layout_params: None,
            lut_config: None,
            build: None,
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

        let (source, parameter_order) = compose_macro_ac_testbench(
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
}
