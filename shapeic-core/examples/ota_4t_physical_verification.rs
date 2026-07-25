#[allow(dead_code)]
#[path = "ota_4t_electrical_verification/analysis.rs"]
mod analysis;

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use analysis::{AcMetrics, PhysicalAnalysis, SizingSummary};
use serde::Deserialize;
use shapeic_layout::PhysicalLookupTable;
use shapeic_lut::verification::{VerificationConfig, VerificationEngine, VerificationInput};
use shapeic_lut::{LookupTable, OperatingPoint};

const TAIL_CURRENT: f64 = 20.0e-6;
const DIFF_LENGTH: f64 = 0.8e-6;
const MIRROR_LENGTH: f64 = 0.4e-6;
const SOURCE_VOLTAGE: f64 = 0.65;
const VOUT_DC: f64 = 1.0;
const VDD_DC: f64 = 1.5;
const VG_DC: f64 = 0.9;
const OUTPUT_BIAS_INDUCTANCE_H: f64 = 1.0e9;
const IHP_MAX_WIDTH_PER_FINGER: f64 = 10.0e-6;
const NMOS_MODEL: &str = "sg13_lv_nmos";
const PMOS_MODEL: &str = "sg13_lv_pmos";
const PHYSICAL_LAYOUT_POLICY: &str = "symmetric-adjacent-with-edge-dummies-v3";
const PEX_PORTS: [&str; 6] = ["VOUT", "VINP", "VINN", "IBIAS", "VDD", "VSS"];

fn main() -> Result<(), Box<dyn Error>> {
    let (nmos_path, pmos_path, physical_path, physical_config) = input_paths()?;
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output_root = unique_output_root(manifest)?;
    let diff_point = OperatingPoint::new(
        DIFF_LENGTH,
        -SOURCE_VOLTAGE,
        VG_DC - SOURCE_VOLTAGE,
        VOUT_DC - SOURCE_VOLTAGE,
    );
    let mirror_point = OperatingPoint::new(MIRROR_LENGTH, 0.0, VOUT_DC - VDD_DC, VOUT_DC - VDD_DC);

    let nmos_table = LookupTable::open(nmos_path)?;
    let pmos_table = LookupTable::open(pmos_path)?;
    let physical_table = PhysicalLookupTable::open(physical_path)?;
    validate_physical_layout_policy(&physical_table)?;
    let physical = analysis::analyze_physical(
        nmos_table.model(NMOS_MODEL)?,
        pmos_table.model(PMOS_MODEL)?,
        &physical_table,
        diff_point,
        mirror_point,
        TAIL_CURRENT / 2.0,
        &manifest.join("examples/ota_4t"),
        &output_root.join("mna"),
    )?;
    validate_finger_width("simplediffpair", physical.diff_pair)?;
    validate_finger_width("currentmirror", physical.current_mirror)?;
    let references = comparison_metrics(physical.metrics)?;
    let pex = generate_pex(
        manifest,
        &output_root,
        &physical_config,
        physical.diff_pair,
        physical.current_mirror,
    )?;

    print_sizing(&physical, diff_point, mirror_point);
    print_pex(&pex);
    let mut input = VerificationInput::new(
        diff_point,
        physical.diff_pair.total_width,
        physical.diff_pair.nf,
    )
    .reference("dc_gain_db", references.dc_gain_db)
    .reference("bandwidth_3db_hz", references.bandwidth_3db_hz);
    if let Some((unity_gain_hz, phase_margin_deg)) = references.loop_metrics() {
        input = input
            .reference("unity_gain_hz", unity_gain_hz)
            .reference("phase_margin_deg", phase_margin_deg);
    } else {
        println!(
            "Shapeic found no unity-gain crossing; UGF and phase margin will not be compared."
        );
    }

    let engine = VerificationEngine::new(verification_config(
        manifest,
        &output_root,
        &pex,
        references.has_loop_metrics(),
    ))?;
    let report = engine.verify_one(&input)?;
    let run = report
        .runs
        .first()
        .ok_or_else(|| io::Error::other("verification produced no run artifacts"))?;
    let shapeic_ac_path = run.artifact_dir.join("shapeic_physical_ac.csv");
    let ngspice_ac_path = run.artifact_dir.join("ngspice_ac.tsv");
    physical.ac_sweep.write_csv(&shapeic_ac_path)?;
    let plot_helper = manifest.join("examples/ota_4t_electrical_verification/plot_ac.py");

    println!();
    println!("Shapeic layout-aware MNA vs. full-layout ngspice PEX");
    print!("{}", report.render_table());
    println!("CSV: {}", report.summary_path.display());
    println!(
        "Rendered NGSpice netlist: {}",
        run.artifact_dir.join("netlist.spice").display()
    );
    println!("Full OTA GDS: {}", pex.gds_path.display());
    println!("Normalized OTA PEX: {}", pex.pex_path.display());
    println!("Raw Magic PEX: {}", pex.raw_pex_path.display());
    println!("Magic script: {}", pex.magic_script_path.display());
    println!("Shapeic physical AC sweep: {}", shapeic_ac_path.display());
    println!("NGSpice PEX AC sweep: {}", ngspice_ac_path.display());
    println!(
        "AC comparison plot: python3 {} {} --shapeic-file {} --title \"4T OTA physical PEX AC comparison\"",
        plot_helper.display(),
        run.artifact_dir.display(),
        shapeic_ac_path
            .file_name()
            .expect("Shapeic AC path has a filename")
            .to_string_lossy(),
    );
    Ok(())
}

#[derive(Clone, Copy)]
struct ComparisonMetrics {
    dc_gain_db: f64,
    bandwidth_3db_hz: f64,
    unity_gain_hz: Option<f64>,
    phase_margin_deg: Option<f64>,
}

impl ComparisonMetrics {
    fn loop_metrics(self) -> Option<(f64, f64)> {
        self.unity_gain_hz.zip(self.phase_margin_deg)
    }

    fn has_loop_metrics(self) -> bool {
        self.loop_metrics().is_some()
    }
}

fn comparison_metrics(metrics: AcMetrics) -> Result<ComparisonMetrics, io::Error> {
    let missing = |metric: &str| {
        io::Error::other(format!(
            "Shapeic physical AC did not find {metric} inside {:.6e}..{:.6e} Hz",
            analysis::AC_MIN_HZ,
            analysis::AC_MAX_HZ
        ))
    };
    if metrics.unity_gain_hz.is_some() != metrics.phase_margin_deg.is_some() {
        return Err(io::Error::other(
            "Shapeic physical AC returned only one of unity-gain frequency and phase margin",
        ));
    }
    Ok(ComparisonMetrics {
        dc_gain_db: metrics.dc_gain_db,
        bandwidth_3db_hz: metrics
            .bandwidth_3db_hz
            .ok_or_else(|| missing("the -3 dB bandwidth"))?,
        unity_gain_hz: metrics.unity_gain_hz,
        phase_margin_deg: metrics.phase_margin_deg,
    })
}

#[derive(Debug, Deserialize)]
struct PexManifest {
    format: String,
    version: u32,
    layout_policy: String,
    cell_name: String,
    subcircuit_name: String,
    port_order: Vec<String>,
    gds_path: PathBuf,
    pex_path: PathBuf,
    raw_pex_path: PathBuf,
    magic_script_path: PathBuf,
    magic_stdout_path: PathBuf,
    magic_stderr_path: PathBuf,
    transistor_count: usize,
    resistor_count: usize,
    capacitor_count: usize,
}

fn generate_pex(
    manifest: &Path,
    output_root: &Path,
    config_path: &Path,
    diff: SizingSummary,
    mirror: SizingSummary,
) -> Result<PexManifest, Box<dyn Error>> {
    let layout_root = manifest.join("../shapeic-layout/lut_generation");
    let python = env::var_os("SHAPEIC_LAYOUT_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let project_python = layout_root.join(".venv-ubuntu/bin/python");
            if project_python.is_file() {
                project_python
            } else {
                PathBuf::from("python3")
            }
        });
    let pex_root = output_root.join("pex");
    let mut command = Command::new(&python);
    command
        .arg(layout_root.join("generate_ota_pex.py"))
        .arg(config_path)
        .arg(&pex_root)
        .arg("--diff-length")
        .arg(format_float(diff.length))
        .arg("--diff-wf")
        .arg(format_float(diff.finger_width))
        .arg("--diff-nf")
        .arg(diff.nf.to_string())
        .arg("--mirror-length")
        .arg(format_float(mirror.length))
        .arg("--mirror-wf")
        .arg(format_float(mirror.finger_width))
        .arg("--mirror-nf")
        .arg(mirror.nf.to_string())
        .current_dir(manifest.join(".."));
    if env::var_os("IHP_PDK_ROOT").is_none() {
        let fallback = env::var_os("PDK_ROOT")
            .map(PathBuf::from)
            .map(|root| root.join("ihp-sg13g2"))
            .unwrap_or_else(|| manifest.join("../../SSTADEX-prev/IHP-Open-PDK/ihp-sg13g2"));
        if fallback.is_dir() {
            command.env("IHP_PDK_ROOT", fallback);
        }
    }
    let output = command.output().map_err(|error| {
        io::Error::other(format!(
            "could not launch OTA PEX generator '{}': {error}",
            python.display()
        ))
    })?;
    fs::create_dir_all(output_root)?;
    fs::write(output_root.join("pex-generator.stdout.log"), &output.stdout)?;
    fs::write(output_root.join("pex-generator.stderr.log"), &output.stderr)?;
    if !output.status.success() {
        let gds_path = pex_root.join("ota_4t.gds");
        let cause = stderr_tail(&output.stderr, 6);
        return Err(io::Error::other(format!(
            "OTA PEX generator exited with {}.\nCause:\n{}\nPreserved GDS: {}\nGenerator stderr: {}",
            output.status,
            cause,
            gds_path.display(),
            output_root.join("pex-generator.stderr.log").display()
        ))
        .into());
    }

    let manifest_path = pex_root.join("manifest.json");
    let pex: PexManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    validate_pex_manifest(&pex)?;
    Ok(pex)
}

fn stderr_tail(stderr: &[u8], line_count: usize) -> String {
    let text = String::from_utf8_lossy(stderr);
    let lines: Vec<_> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let start = lines.len().saturating_sub(line_count);
    if start == lines.len() {
        "(no stderr output)".to_owned()
    } else {
        lines[start..].join("\n")
    }
}

fn validate_pex_manifest(pex: &PexManifest) -> Result<(), io::Error> {
    if pex.format != "shapeic-ota-pex" || pex.version != 1 {
        return Err(io::Error::other(format!(
            "unsupported OTA PEX manifest '{}', version {}",
            pex.format, pex.version
        )));
    }
    if pex.layout_policy != PHYSICAL_LAYOUT_POLICY {
        return Err(io::Error::other(format!(
            "OTA PEX uses layout policy '{}', expected '{}'",
            pex.layout_policy, PHYSICAL_LAYOUT_POLICY
        )));
    }
    if pex.port_order.iter().map(String::as_str).ne(PEX_PORTS) {
        return Err(io::Error::other(format!(
            "OTA PEX ports are {:?}, expected {:?}",
            pex.port_order, PEX_PORTS
        )));
    }
    for (name, path) in [
        ("GDS", &pex.gds_path),
        ("PEX", &pex.pex_path),
        ("raw PEX", &pex.raw_pex_path),
        ("Magic script", &pex.magic_script_path),
        ("Magic stdout", &pex.magic_stdout_path),
        ("Magic stderr", &pex.magic_stderr_path),
    ] {
        if !path.is_file() {
            return Err(io::Error::other(format!(
                "OTA {name} artifact does not exist: {}",
                path.display()
            )));
        }
    }
    if pex.cell_name.is_empty()
        || pex.subcircuit_name.is_empty()
        || pex.transistor_count < 4
        || pex.capacitor_count == 0
    {
        return Err(io::Error::other(
            "OTA PEX manifest has invalid topology counts",
        ));
    }
    Ok(())
}

fn verification_config(
    manifest: &Path,
    output_root: &Path,
    pex: &PexManifest,
    include_loop_metrics: bool,
) -> VerificationConfig {
    let sstadex_root = manifest.join("../../SSTADEX-prev");
    let pdk_root = env::var_os("PDK_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| sstadex_root.join("IHP-Open-PDK"));
    let ngspice = pdk_root.join("ihp-sg13g2/libs.tech/ngspice");

    VerificationConfig::new(
        manifest.join("examples/ota_4t_physical_verification/ihp_sg13g2_pex.spice"),
        output_root.join("verification"),
    )
    .max_width_per_finger(IHP_MAX_WIDTH_PER_FINGER)
    .template_variable(
        "model_library",
        ngspice.join("models/cornerMOSlv.lib").display().to_string(),
    )
    .template_variable(
        "psp_osdi",
        ngspice.join("osdi/psp103.osdi").display().to_string(),
    )
    .template_variable(
        "psp_nqs_osdi",
        ngspice.join("osdi/psp103_nqs.osdi").display().to_string(),
    )
    .template_variable("pex_path", pex.pex_path.display().to_string())
    .template_variable("pex_subcircuit", pex.subcircuit_name.clone())
    .template_variable("tail_current", format_float(TAIL_CURRENT))
    .template_variable("vdd_dc", format_float(VDD_DC))
    .template_variable("vg_dc", format_float(VG_DC))
    .template_variable("vout_dc", format_float(VOUT_DC))
    .template_variable(
        "output_bias_inductance",
        format_float(OUTPUT_BIAS_INDUCTANCE_H),
    )
    .template_variable(
        "ac_points_per_decade",
        analysis::AC_POINTS_PER_DECADE.to_string(),
    )
    .template_variable("ac_min_hz", format_float(analysis::AC_MIN_HZ))
    .template_variable("ac_max_hz", format_float(analysis::AC_MAX_HZ))
    .template_variable(
        "optional_ac_measurements",
        optional_ac_measurements(include_loop_metrics),
    )
    .template_variable(
        "optional_result_vectors",
        if include_loop_metrics {
            " unity_gain_hz phase_margin_deg"
        } else {
            ""
        },
    )
}

fn optional_ac_measurements(include_loop_metrics: bool) -> &'static str {
    if include_loop_metrics {
        "meas ac unity_gain_hz when gain_db=0 fall=1\n\
         meas ac phase_at_unity find phase_deg at=unity_gain_hz\n\
         let phase_margin_deg = 180 + phase_at_unity"
    } else {
        "* Shapeic found no unity-gain crossing; UGF and phase margin are omitted."
    }
}

fn validate_finger_width(primitive: &str, sizing: SizingSummary) -> Result<(), io::Error> {
    if sizing.finger_width >= IHP_MAX_WIDTH_PER_FINGER {
        return Err(io::Error::other(format!(
            "{primitive} Wf={:.6e} must be less than {:.6e}",
            sizing.finger_width, IHP_MAX_WIDTH_PER_FINGER
        )));
    }
    Ok(())
}

fn validate_physical_layout_policy(table: &PhysicalLookupTable) -> Result<(), io::Error> {
    let found = &table.metadata().layout_policy;
    if found == PHYSICAL_LAYOUT_POLICY {
        return Ok(());
    }
    Err(io::Error::other(format!(
        "OTA physical LUT uses layout policy '{found}', expected \
         '{PHYSICAL_LAYOUT_POLICY}'. Regenerate it after the multifinger terminal-bus correction"
    )))
}

fn print_sizing(
    physical: &PhysicalAnalysis,
    diff_point: OperatingPoint,
    mirror_point: OperatingPoint,
) {
    println!("Physical OTA verification point");
    println!(
        "{:<16} | {:>9} | {:>9} | {:>9} | {:>9} | {:>5} | {:>11} | {:>10} | {:>10} | {:>10}",
        "primitive",
        "L [um]",
        "VGS [V]",
        "VDS [V]",
        "Wf [um]",
        "nf",
        "W total [um]",
        "Ireq [uA]",
        "Id [uA]",
        "error [%]",
    );
    println!(
        "-----------------+-----------+-----------+-----------+-----------+-------+-------------+------------+------------+-----------"
    );
    print_sizing_row("simplediffpair", physical.diff_pair, diff_point);
    print_sizing_row("currentmirror", physical.current_mirror, mirror_point);
}

fn print_sizing_row(name: &str, sizing: SizingSummary, point: OperatingPoint) {
    println!(
        "{name:<16} | {:>9.3} | {:>9.3} | {:>9.3} | {:>9.3} | {:>5} | {:>11.3} | {:>10.3} | {:>10.3} | {:>+10.3e}",
        sizing.length * 1.0e6,
        point.vgs,
        point.vds,
        sizing.finger_width * 1.0e6,
        sizing.nf,
        sizing.total_width * 1.0e6,
        sizing.requested_current * 1.0e6,
        sizing.predicted_current * 1.0e6,
        sizing.relative_error_percent()
    );
}

fn print_pex(pex: &PexManifest) {
    println!(
        "Full OTA PEX: {} MOS, {} R, {} C; subcircuit {}",
        pex.transistor_count, pex.resistor_count, pex.capacitor_count, pex.subcircuit_name
    );
}

fn input_paths() -> Result<(PathBuf, PathBuf, PathBuf, PathBuf), io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_4t_physical_verification"));
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "usage: {} <nmos-5d.npz> <pmos-5d.npz> <physical.npz> <physical-config.toml>",
                Path::new(&executable).display()
            ),
        )
    };
    let nmos = arguments.next().ok_or_else(&usage)?;
    let pmos = arguments.next().ok_or_else(&usage)?;
    let physical = arguments.next().ok_or_else(&usage)?;
    let config = arguments.next().ok_or_else(&usage)?;
    if arguments.next().is_some() {
        return Err(usage());
    }
    Ok((nmos.into(), pmos.into(), physical.into(), config.into()))
}

fn unique_output_root(manifest: &Path) -> Result<PathBuf, std::time::SystemTimeError> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    Ok(manifest
        .join("../target/shapeic-ota-4t-physical-verification")
        .join(timestamp.to_string()))
}

fn format_float(value: f64) -> String {
    format!("{value:.17e}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_diff_pair_uses_vss_bulk() {
        let point = OperatingPoint::new(
            DIFF_LENGTH,
            -SOURCE_VOLTAGE,
            VG_DC - SOURCE_VOLTAGE,
            VOUT_DC - SOURCE_VOLTAGE,
        );
        assert!((point.vbs + 0.65).abs() < 1.0e-12);
        assert!((point.vgs - 0.25).abs() < 1.0e-12);
        assert!((point.vds - 0.35).abs() < 1.0e-12);
    }

    #[test]
    fn generator_error_reports_only_the_stderr_tail() {
        let stderr = b"one\ntwo\nthree\nfour\n";
        assert_eq!(stderr_tail(stderr, 2), "three\nfour");
        assert_eq!(stderr_tail(b"", 2), "(no stderr output)");
    }

    #[test]
    fn rejects_invalid_pex_port_order() {
        let pex = PexManifest {
            format: "shapeic-ota-pex".to_owned(),
            version: 1,
            layout_policy: PHYSICAL_LAYOUT_POLICY.to_owned(),
            cell_name: "ota".to_owned(),
            subcircuit_name: "ota_flat".to_owned(),
            port_order: vec!["VINP".to_owned()],
            gds_path: PathBuf::new(),
            pex_path: PathBuf::new(),
            raw_pex_path: PathBuf::new(),
            magic_script_path: PathBuf::new(),
            magic_stdout_path: PathBuf::new(),
            magic_stderr_path: PathBuf::new(),
            transistor_count: 8,
            resistor_count: 0,
            capacitor_count: 1,
        };
        assert!(validate_pex_manifest(&pex).is_err());
    }

    #[test]
    fn physical_verification_template_is_fully_bound() {
        let pex = PexManifest {
            format: "shapeic-ota-pex".to_owned(),
            version: 1,
            layout_policy: PHYSICAL_LAYOUT_POLICY.to_owned(),
            cell_name: "ota".to_owned(),
            subcircuit_name: "ota_flat".to_owned(),
            port_order: PEX_PORTS.iter().map(|port| (*port).to_owned()).collect(),
            gds_path: PathBuf::from("ota.gds"),
            pex_path: PathBuf::from("ota.pex.spice"),
            raw_pex_path: PathBuf::from("ota.raw.pex.spice"),
            magic_script_path: PathBuf::from("ota.tcl"),
            magic_stdout_path: PathBuf::from("magic.stdout"),
            magic_stderr_path: PathBuf::from("magic.stderr"),
            transistor_count: 8,
            resistor_count: 0,
            capacitor_count: 1,
        };
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let config = verification_config(manifest, Path::new("unused-output"), &pex, false);
        VerificationEngine::new(config).expect("valid physical verification template");
        let config = verification_config(manifest, Path::new("unused-output"), &pex, true);
        VerificationEngine::new(config)
            .expect("valid physical verification template with loop metrics");
    }
}
