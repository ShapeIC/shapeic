use serde::de::Error;
use shapeic_core::macro_model::MacroAcTestbench;
use std::ffi::OsString;
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::circuit::Circuit;
use shapeic_core::macro_model::{MacroExecutionConfig, Macro, MacroPort, MacroPortRole, MacroCompactSeed};

use shapeic_lut::LookupTable;
use shapeic_core::testbench::{AcAnalysis, TransferFunction, TransferPolarity};
use shapeic_core::analysis::{AdaptiveAcConfig, AdaptiveAcPolicy, AnalysisMode, AnalysisTargets, AcMetricSet};

const COMMON_SOURCE_INSTANCE: &str = "xcs";
const OTA_1STAGE_INSTANCE: &str = "xota_1stage";
const OTA_1STAGE: &str = "ota_1stage"; 
const OTA_2STAGE: &str = "ota_2stage";
const OTA_2STAGE_TB: &str = "ota_2stage_gain";

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
    vbias_start: f64,
    vbias_stop: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = cli_options();
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


}

fn ota_1stage(spec: PdkSpec, testbench: PathBuf) -> Macro {
    Macro::new(
        OTA_1STAGE,
        ota_1stage_ports(),
        Circuit::default(),
        ota_1stage_compact_model()
    )
    .with_compact_seed(ota_seed(spec))
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
            ],
        )
        .build();
    Macro::new(OTA_2STAGE, Vec::new(), circuit, ota_2stage_compact_model())
        .with_ac_testbench(MacroAcTestbench::from_spice_file(
            OTA_2STAGE_TB,
            testbench,
            ac_analysis()
        ))
}

fn ota_1stage_ports() -> Vec<MacroPort> {
    vec![
        MacroPort::new("VINP", MacroPortRole::Input),
        MacroPort::new("VINN", MacroPortRole::Input),
        MacroPort::new("VOUT", MacroPortRole::Output),
        MacroPort::new("IBIAS", MacroPortRole::Bias),
        MacroPort::new("VDD", MacroPortRole::Supply),
    ]
}

fn ota_1stage_compact_model() -> Circuit {
    Circuit::builder()
        .vccs("gm_dp_m1", "VOUT", "IBIAS", "VINP", "IBIAS", "gm_dp")
        .resistor("ro_dp_m1", "VOUT", "IBIAS", "ro_dp")
        .vccs("gm_dp_m2", "N1", "IBIAS", "VINN", "IBIAS", "gm_dp")
        .resistor("ro_dp_m2", "N1", "IBIAS", "ro_dp")
        .vccs("gm_cm_m1", "VOUT", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m1", "VOUT", "VDD", "ro_cm")
        .vccs("gm_cm_m2", "N1", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m2", "N1", "VDD", "ro_cm")
        .capacitor("c_out", "VOUT", "IBIAS", 1.0e-12)
        .build()
}

fn ota_2stage_compact_model() -> Circuit {
    Circuit::builder()
        .vccs("gm_dp_m1", "VOUT", "IBIAS", "VINP", "IBIAS", "gm_dp")
        .resistor("ro_dp_m1", "VOUT", "IBIAS", "ro_dp")
        .vccs("gm_dp_m2", "N1", "IBIAS", "VINN", "IBIAS", "gm_dp")
        .resistor("ro_dp_m2", "N1", "IBIAS", "ro_dp")
        .vccs("gm_cm_m1", "VOUT", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m1", "VOUT", "VDD", "ro_cm")
        .vccs("gm_cm_m2", "N1", "VDD", "N1", "VDD", "gm_cm")
        .resistor("ro_cm_m2", "N1", "VDD", "ro_cm")
        .capacitor("c_out", "VOUT", "IBIAS", 1.0e-12)
        .build()
}

fn ota_seed(spec: PdkSpec) -> MacroCompactSeed {
    MacroCompactSeed::new(
        [
            ("gm_dp", 1.0e-3),
            ("ro_dp", 100.0e3),
            ("gm_cm", 1.0e-3),
            ("ro_cm", 100.0e3),
        ],
        [
            ("VINP", spec.vin),
            ("VINN", spec.vin),
            ("VOUT", spec.vout_start),
            ("IBIAS", spec.vbias_start),
            ("VDD", spec.vdd),
        ],
    )
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
