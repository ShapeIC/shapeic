use std::error::Error;
use shapeic_core::macro_model::MacroExploration;
use shapeic_core::macro_model::explore_macro_hierarchy;
use std::collections::HashMap;
use shapeic_core::macro_model::MacroAcTestbench;
use shapeic_core::macro_model::MacroDesignVariable;
use shapeic_core::macro_model::MacroHierarchyExplorationInput;
use shapeic_core::macro_model::MacroPrimitiveDefault;
use shapeic_core::macro_model::MacroPublicInputAlias;
use shapeic_core::macro_model::MacroSpecification;
use std::ffi::OsString;
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::circuit::Circuit;
use shapeic_core::macro_model::{MacroExecutionConfig, Macro, MacroPort, MacroPortRole, MacroCompactSeedSet, MacroSpecificationBounds, MacroSpecificationSource, MacroCatalog, MacroHierarchyExplorationResult, MacroHierarchyMode, MacroHierarchyPathInput, MacroHierarchyPath, MacroHierarchyRetentionPolicy};
use shapeic_core::primitive::build::{PrimitiveBuildInputKind, PrimitiveBuildInput, PrimitiveBuildValue};

use shapeic_lut::LookupTable;
use shapeic_core::testbench::{AcAnalysis, TransferFunction, TransferPolarity};
use shapeic_core::analysis::{AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets, AcMetricSet, AcMetric};
use shapeic_core::utils::linspace;

const COMMON_SOURCE_INSTANCE: &str = "xcs";
const OTA_1STAGE_INSTANCE: &str = "xota_1stage";
const OTA_1STAGE_TB: &str = "ota_1stage_gain";
const OTA_1STAGE: &str = "ota_1stage"; 
const OTA_2STAGE: &str = "ota_2stage";
const OTA_2STAGE_TB: &str = "ota_2stage_gain";
const GAIN_SPECIFICATION: &str = "dc_gain_db";

const VOUT_POINTS: usize = 5;
const VOUT_1STAGE_POINTS: usize = 3;

const MIN_DC_GAIN_DB: f64 = 25.0;
const MIN_BANDWIDTH_3DB_HZ: f64 = 1.0e6;
const MIN_UNITY_GAIN_HZ: f64 = 1.0e7;
const MIN_PHASE_MARGIN_DEG: f64 = 45.0;

#[derive(Clone, Copy, Debug, PartialEq)]
struct PdkSpec {
    label: &'static str,
    pdk: &'static str,
    nmos_model: &'static str,
    pmos_model: &'static str,
    tail_current: f64,
    mirror_reference: f64,
    vout_start: f64,
    vout_stop: f64,
    vdd: f64,
    vin: f64,
    vout_1stage_start: f64,
    vout_1stage_stop: f64,
}

const IHP_SPEC: PdkSpec = PdkSpec {
    label: "IHP SG13G2",
    pdk: "ihp-sg13g2",
    nmos_model: "sg13_lv_nmos",
    pmos_model: "sg13_lv_pmos",
    tail_current: 20.0e-6,
    mirror_reference: 0.9,
    vout_start: 0.95,
    vout_stop: 1.1,
    vdd: 1.5,
    vin: 0.9,
    vout_1stage_start: 0.95,
    vout_1stage_stop: 1.1,
};

fn main() -> Result<(), Box<dyn Error>> {
    let options = cli_options()?;
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let testbench = manifest.join("examples/ota_2stage/gain.spice");
    let primitive_catalog = load_primitive_catalog(&manifest.join("../shapeic-cellkit/primitives"))
        .map_err(|error| format!("{error:?}"))?;

    let nmos_table = LookupTable::open(&options.nmos_path)?;
    let pmos_table = LookupTable::open(&options.pmos_path)?;
    let spec = resolve_pdk_spec(&nmos_table, &pmos_table)?;
    let nmos = nmos_table.model(spec.nmos_model)?;
    let pmos = pmos_table.model(spec.pmos_model)?;

    let ota_1stage = ota_1stage(spec, testbench.clone());
    let ota_2stage = ota_2stage(spec, testbench.clone());
    let macro_catalog = MacroCatalog::from_macros([ota_1stage, ota_2stage])?;

    let mut input = MacroHierarchyExplorationInput::new();
    input.register_device_model("nmos", nmos)?;
    input.register_device_model("pmos", pmos)?;
    input.register_path(MacroHierarchyPath::root(OTA_2STAGE)?, top_path_input(spec))?;
    input.set_retention_policy(
        MacroHierarchyRetentionPolicy::FullPreviews,
    );
    input.set_execution_config(options.execution_config()?);

    let result = explore_macro_hierarchy(OTA_2STAGE, &macro_catalog, &primitive_catalog, &input)?;

    println!("{} hierarchical four-transistor OTA", spec.label);
    print_hierarchy(&result);
    print_derivations(&result);

    Ok(())
}

fn ota_1stage(spec: PdkSpec, testbench: PathBuf) -> Macro {
    Macro::new(
        OTA_1STAGE,
        ota_1stage_ports(),
        Circuit::default(),
        ota_1stage_compact_model()
    )
    .with_hierarchy_mode(MacroHierarchyMode::BlackBox)
    .with_compact_seeds(ota_1stage_seed(spec))
    .with_design_variable(MacroDesignVariable::new(
        "vout",
        PrimitiveBuildInputKind::Vector,
        Vec::new(),
    ))
}

fn ota_2stage(spec: PdkSpec, testbench: PathBuf) -> Macro {
    let circuit = Circuit::builder()
        .primitive(
            COMMON_SOURCE_INSTANCE,
            "simplecommonsource",
            [
                ("VIN", "VOUT_1STAGE"),
                ("VOUT", "VOUT"),
                ("VDD", "VDD")
            ]
        )
        .macro_instance(
            OTA_1STAGE_INSTANCE,
            OTA_1STAGE,
            [
                ("VINP", "VINP"),
                ("VINN", "VINN"),
                ("VOUT", "VOUT_1STAGE"),
                ("IBIAS", "IBIAS"),
                ("VDD", "VDD"),
                ("VSS", "VSS")
            ],
        )
        .build();
    Macro::new(OTA_2STAGE, Vec::new(), circuit, ota_2stage_compact_model())
        .with_ac_testbench(MacroAcTestbench::from_spice_file(
            OTA_2STAGE_TB,
            testbench,
            ac_analysis()
        ))
        .with_specification(MacroSpecification::new(
            GAIN_SPECIFICATION,
            MacroSpecificationSource::ac_metric(OTA_2STAGE_TB, AcMetric::DcGainDb),
            MacroSpecificationBounds::at_least(MIN_DC_GAIN_DB),
        ))
        .with_primitive_default(MacroPrimitiveDefault::new(
            COMMON_SOURCE_INSTANCE, 
            common_source_input(spec),
            Vec::new()
        ))
        .with_design_variable(MacroDesignVariable::new(
            "vout_1stage",
            PrimitiveBuildInputKind::Vector,
            Vec::new(),
        ))
        .with_public_input_alias(MacroPublicInputAlias::new(
            "vout_1stage", 
            OTA_1STAGE_INSTANCE, 
            "vout", 
            "VOUT"
        ))
}

fn top_path_input(spec: PdkSpec) -> MacroHierarchyPathInput {
    let mut input = MacroHierarchyPathInput::new();
    input
        .register_design_variable_override(
            "vout_1stage",
            PrimitiveBuildValue::Vector(linspace(spec.vout_1stage_start, spec.vout_1stage_stop, VOUT_1STAGE_POINTS)),
        )
        .expect("top-level VOUT_1STAGE is registered once");
    input
}

fn ota_1stage_ports() -> Vec<MacroPort> {
    vec![
        MacroPort::new("VINP", MacroPortRole::Input),
        MacroPort::new("VINN", MacroPortRole::Input),
        MacroPort::new("VOUT", MacroPortRole::Output),
        MacroPort::new("IBIAS", MacroPortRole::Bias),
        MacroPort::new("VDD", MacroPortRole::Supply),
        MacroPort::new("VSS", MacroPortRole::Ground),
    ]
}

fn ota_1stage_compact_model() -> Circuit {
    Circuit::builder()
        .vccs("gm_ota", "VOUT", "VSS", "VINP", "VSS", "gm_ota")
        .resistor("ro_ota", "VOUT", "VSS", "ro_ota")
        .capacitor("c_ota", "VOUT", "VSS", "c_ota")
        .build()
}

fn ota_2stage_compact_model() -> Circuit {
    Circuit::builder()
        .vccs("gm_ota", "VOUT", "VSS", "VINP", "VSS", 1.0)
        .resistor("ro_ota", "VOUT", "VSS", 1.0)
        .capacitor("c_ota", "VOUT", "VSS", 1.0)
        .build()
}

fn ota_1stage_seed(spec: PdkSpec) -> MacroCompactSeedSet {
    MacroCompactSeedSet::aligned([
        ("gm_ota", vec![0.7e-3, 1.0e-3, 1.3e-3, 1e-3]),
        ("ro_ota", vec![140.0e3, 100.0e3, 75.0e3, 1e5]),
        ("c_ota", vec![1.0e-12, 2.0e-12, 3.0e-12, 1e-12]),
    ])
}

fn ac_analysis() -> AcAnalysis {
    AcAnalysis::new(
        TransferFunction::new("VINP", "VOUT").with_polarity(TransferPolarity::Negative),
        AdaptiveAcConfig {
            min_frequency_hz: 1.0,
            max_frequency_hz: 100.0e9,
            coarse_points_per_decade: 4,
            crossing_relative_tolerance: 0.005,
            max_refinement_steps: 32,
            retain_samples: false,
        },
        AdaptiveAcPolicy {
            mode: AnalysisMode::Prune,
            targets: AnalysisTargets {
                min_dc_gain_db: None,
                min_bandwidth_3db_hz: Some(MIN_BANDWIDTH_3DB_HZ),
                min_unity_gain_hz: Some(MIN_UNITY_GAIN_HZ),
                min_phase_margin_deg: Some(MIN_PHASE_MARGIN_DEG),
            },
            metrics: AcMetricSet::ALL,
        },
    )
}

fn common_source_input(spec: PdkSpec) -> PrimitiveBuildInput {
    PrimitiveBuildInput::new(HashMap::from([
        (
            "current".to_owned(),
            PrimitiveBuildValue::Scalar(spec.tail_current),
        ),
        (
            "VIN".to_owned(), 
            PrimitiveBuildValue::Vector(linspace(spec.vout_1stage_start, spec.vout_1stage_stop, VOUT_1STAGE_POINTS)),
        ),
        (
            "VOUT".to_owned(),
            PrimitiveBuildValue::Vector(linspace(spec.vout_start, spec.vout_stop, VOUT_POINTS)),
        ),
        (
            "VDD".to_owned(), 
            PrimitiveBuildValue::Scalar(spec.vdd)),
    ]))
}

fn resolve_pdk_spec(nmos: &LookupTable, pmos: &LookupTable) -> Result<PdkSpec, io::Error> {
    let nmos_spec = identify_pdk(nmos, true)?;
    let pmos_spec = identify_pdk(pmos, false)?;
    if nmos_spec != pmos_spec {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "NMOS LUT uses PDK '{}', but PMOS LUT uses '{}'",
                nmos_spec.pdk, pmos_spec.pdk
            ),
        ));
    }
    Ok(nmos_spec)
}

fn identify_pdk(table: &LookupTable, nmos: bool) -> Result<PdkSpec, io::Error> {
    let candidates = [IHP_SPEC];
    if let Some(pdk) = table.pdk() {
        return candidates
            .into_iter()
            .find(|spec| spec.pdk == pdk)
            .ok_or_else(|| unsupported_pdk(pdk));
    }
    let names = table.model_names().collect::<Vec<_>>();
    candidates
        .into_iter()
        .find(|spec| {
            names.contains(&if nmos {
                spec.nmos_model
            } else {
                spec.pmos_model
            })
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "LUT metadata and model names do not identify a supported PDK",
            )
        })
}
fn unsupported_pdk(pdk: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("unsupported LUT PDK '{pdk}'"),
    )
}

fn cli_options() -> Result<CliOptions, io::Error> {
let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_2stage"));
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
            "usage: {} <nmos-5d.npz> <pmos-5d.npz> [--workers N] [--batch-size N]",
                Path::new(&executable).display()
            ),
        )
    };
    let mut paths = Vec::new();
    let mut workers = None;
    let mut batch_size = None;
    while let Some(argument) = arguments.next() {
        if argument == "--workers" {
            workers = Some(parse_usize(
                arguments.next().ok_or_else(&usage)?,
                "--workers",
            )?);
        } else if argument == "--batch-size" {
            batch_size = Some(parse_usize(
                arguments.next().ok_or_else(&usage)?,
                "--batch-size",
            )?);
        } else if argument.to_string_lossy().starts_with("--") {
            return Err(usage());
        } else {
            paths.push(PathBuf::from(argument));
        }
    }
    if paths.len() != 2 {
        return Err(usage());
    }
    let options = CliOptions {
        nmos_path: paths.remove(0),
        pmos_path: paths.remove(0),
        workers,
        batch_size,
    };
    options.execution_config()?;
    Ok(options)
}

fn parse_usize(value: OsString, flag: &str) -> Result<usize, io::Error> {
    value
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "argument is not UTF-8"))?
        .parse::<usize>()
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{flag} requires a positive integer"),
            )
        })
}

#[derive(Debug)]
struct CliOptions {
    nmos_path: PathBuf,
    pmos_path: PathBuf,
    workers: Option<usize>,
    batch_size: Option<usize>,
}

impl CliOptions {
    fn execution_config(&self) -> Result<MacroExecutionConfig, io::Error> {
        let mut execution = MacroExecutionConfig::sequential();
        if let Some(workers) = self.workers {
            execution = execution
                .with_parallel_candidate_build(workers)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
                .with_parallel_electrical_analysis(workers)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        }
        if let Some(batch_size) = self.batch_size {
            execution = execution
                .with_electrical_batch_size(batch_size)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        }
        Ok(execution)
    }
}

fn print_hierarchy(result: &MacroHierarchyExplorationResult) {
    println!("\nHierarchy execution");
    for (path, node) in result.nodes() {
        let accepted = node.result().map_or(0, |result| result.accepted().len());
        println!(
            "  {path}: macro={}, status={:?}, accepted={accepted}",
            node.macro_name(),
            node.status()
        );
    }
    let statistics = result.statistics();
    println!(
        "  paths={}, previews={}, definitive={}, frequency_evaluations={}, total={:?}",
        statistics.total_paths(),
        statistics.previews_executed(),
        statistics.definitive_evaluations(),
        statistics.frequency_evaluations(),
        statistics.total_duration(),
    );
}

fn print_derivations(result: &MacroHierarchyExplorationResult) {
    println!("\nParent-to-child derivations");
    for (_, derivation) in result.derivations() {
        println!(
            "  {} -> {}",
            derivation.parent_path(),
            derivation.child_path()
        );
        for entry in derivation.conditions().audit() {
            println!(
                "    {}: {:?} over {} row(s) => {:?}; {:?}",
                entry.expression(),
                entry.reduction(),
                entry.source_rows(),
                entry.reduced_value(),
                entry.effective_condition(),
            );
        }
        for entry in derivation.conditions().public_input_audit() {
            println!(
                "    input {} via {}.{} -> {}: {:?} over {} row(s); {:?}",
                entry.parent_variable(),
                entry.child_instance(),
                entry.interface_port(),
                entry.child_variable(),
                entry.values(),
                entry.source_rows(),
                entry.effective_condition(),
            );
        }
    }
}


