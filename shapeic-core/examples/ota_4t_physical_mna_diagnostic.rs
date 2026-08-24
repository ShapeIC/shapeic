#[allow(dead_code)]
#[path = "ota_4t_electrical_verification/analysis.rs"]
mod analysis;

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use analysis::{AcMetrics, OTA_INTERCONNECT_NODES, OtaInterconnectMatrices};
use ndarray::Array2;
use ndarray_npy::NpzReader;
use serde::Deserialize;
use shapeic_layout::PhysicalLookupTable;
use shapeic_lut::{LookupTable, OperatingPoint};

const TAIL_CURRENT: f64 = 20.0e-6;
const DIFF_LENGTH: f64 = 0.8e-6;
const MIRROR_LENGTH: f64 = 0.4e-6;
const SOURCE_VOLTAGE: f64 = 0.65;
const VOUT_DC: f64 = 1.0;
const VDD_DC: f64 = 1.5;
const VG_DC: f64 = 0.9;
const NMOS_MODEL: &str = "sg13_lv_nmos";
const PMOS_MODEL: &str = "sg13_lv_pmos";
const PHYSICAL_LAYOUT_POLICY: &str = "symmetric-adjacent-with-edge-dummies-v3";

fn main() -> Result<(), Box<dyn Error>> {
    let (nmos_path, pmos_path, physical_path, diagnostic_root) = input_paths()?;
    let summary = load_summary(&diagnostic_root.join("summary.json"))?;
    let interconnect = load_interconnect(&summary.artifacts.matrices)?;
    let physical = PhysicalLookupTable::open(physical_path)?;
    if physical.metadata().format_version != 2
        || physical.metadata().layout_policy != PHYSICAL_LAYOUT_POLICY
    {
        return Err(io::Error::other(
            "the MNA diagnostic requires the current physical LUT v2 layout policy",
        )
        .into());
    }
    let nmos = LookupTable::open(nmos_path)?;
    let pmos = LookupTable::open(pmos_path)?;
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output_root = unique_output_root(manifest)?;
    let result = analysis::analyze_physical_mna_diagnostic(
        nmos.model(NMOS_MODEL)?,
        pmos.model(PMOS_MODEL)?,
        &physical,
        OperatingPoint::new(
            DIFF_LENGTH,
            0.0,
            VG_DC - SOURCE_VOLTAGE,
            VOUT_DC - SOURCE_VOLTAGE,
        ),
        OperatingPoint::new(MIRROR_LENGTH, 0.0, VOUT_DC - VDD_DC, VOUT_DC - VDD_DC),
        TAIL_CURRENT / 2.0,
        "IBIAS",
        &interconnect,
        &manifest.join("examples/ota_4t"),
        &output_root.join("mna"),
    )?;

    println!("Physical MNA capacitance diagnostic");
    println!(
        "simplediffpair: Wf={:.9} um, nf={}; currentmirror: Wf={:.9} um, nf={}",
        result.diff_pair.finger_width * 1.0e6,
        result.diff_pair.nf,
        result.current_mirror.finger_width * 1.0e6,
        result.current_mirror.nf,
    );
    println!(
        "{:<30} {:>14} {:>14} {:>14} {:>14}",
        "variant", "gain [dB]", "f3dB [Hz]", "UGF [Hz]", "PM [deg]"
    );
    print_metrics("current physical LUT", result.lut_current);
    print_metrics("exact local bulk binding", result.exact_bound_local);
    print_metrics("exact local + OTA routing", result.full_ota_interconnect);
    println!("MNA artifacts: {}", output_root.display());
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RoutingSummary {
    format: String,
    version: u32,
    nodes: Vec<String>,
    diff_bulk_node: String,
    artifacts: RoutingArtifacts,
}

#[derive(Debug, Deserialize)]
struct RoutingArtifacts {
    matrices: PathBuf,
}

fn load_summary(path: &Path) -> Result<RoutingSummary, Box<dyn Error>> {
    let summary: RoutingSummary = serde_json::from_reader(File::open(path)?)?;
    if summary.format != "shapeic-ota-routing-diagnostic"
        || summary.version != 1
        || summary
            .nodes
            .iter()
            .map(String::as_str)
            .ne(OTA_INTERCONNECT_NODES)
        || summary.diff_bulk_node != "IBIAS"
        || !summary.artifacts.matrices.is_file()
    {
        return Err(io::Error::other(
            "routing diagnostic is incompatible with the source-tied OTA MNA point",
        )
        .into());
    }
    Ok(summary)
}

fn load_interconnect(path: &Path) -> Result<OtaInterconnectMatrices, Box<dyn Error>> {
    let mut archive = NpzReader::new(File::open(path)?)?;
    let exact_bound_conductance: Array2<f64> = archive.by_name("exact_bound_local_g")?;
    let exact_bound_capacitance: Array2<f64> = archive.by_name("exact_bound_local_c")?;
    let full_ota_conductance: Array2<f64> = archive.by_name("full_ota_g")?;
    let full_ota_capacitance: Array2<f64> = archive.by_name("full_ota_c")?;
    Ok(OtaInterconnectMatrices::new(
        exact_bound_conductance,
        exact_bound_capacitance,
        full_ota_conductance,
        full_ota_capacitance,
    )?)
}

fn print_metrics(name: &str, metrics: AcMetrics) {
    println!(
        "{name:<30} {:>14} {:>14} {:>14} {:>14}",
        metric(metrics.dc_gain_db),
        metric(metrics.bandwidth_3db_hz),
        metric(metrics.unity_gain_hz),
        metric(metrics.phase_margin_deg),
    );
}

fn metric(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6e}"))
        .unwrap_or_else(|| "-".to_owned())
}

fn input_paths() -> Result<(PathBuf, PathBuf, PathBuf, PathBuf), io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_4t_physical_mna_diagnostic"));
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "usage: {} <nmos-5d.npz> <pmos-5d.npz> <physical-v2.npz> \
                 <routing-diagnostic-directory>",
                Path::new(&executable).display()
            ),
        )
    };
    let nmos = arguments.next().ok_or_else(&usage)?;
    let pmos = arguments.next().ok_or_else(&usage)?;
    let physical = arguments.next().ok_or_else(&usage)?;
    let diagnostic = arguments.next().ok_or_else(&usage)?;
    if arguments.next().is_some() {
        return Err(usage());
    }
    Ok((nmos.into(), pmos.into(), physical.into(), diagnostic.into()))
}

fn unique_output_root(manifest: &Path) -> Result<PathBuf, std::time::SystemTimeError> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    Ok(manifest
        .join("../target/shapeic-ota-4t-physical-mna-diagnostic")
        .join(timestamp.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_finite_or_wrong_size_interconnect_matrices() {
        let valid = Array2::zeros((6, 6));
        assert!(
            OtaInterconnectMatrices::new(
                valid.clone(),
                valid.clone(),
                valid.clone(),
                valid.clone(),
            )
            .is_ok()
        );
        let invalid = Array2::zeros((5, 5));
        assert!(
            OtaInterconnectMatrices::new(invalid, valid.clone(), valid.clone(), valid).is_err()
        );
    }
}
