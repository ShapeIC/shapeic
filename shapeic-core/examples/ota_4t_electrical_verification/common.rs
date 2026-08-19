use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use shapeic_lut::verification::VerificationConfig;

use super::analysis;
use super::analysis::SizingSummary;

pub const TAIL_CURRENT: f64 = 20.0e-6;
pub const VOUT_DC: f64 = 1.1;
pub const VDD_DC: f64 = 1.5;
pub const VG_DC: f64 = 0.9;
pub const OUTPUT_BIAS_INDUCTANCE_H: f64 = 1.0e9;
pub const IHP_MAX_WIDTH_PER_FINGER: f64 = 10.0e-6;
pub const NMOS_MODEL: &str = "sg13_lv_nmos";
pub const PMOS_MODEL: &str = "sg13_lv_pmos";

pub fn verification_config(
    manifest: &Path,
    output_root: &Path,
    mirror: SizingSummary,
    include_loop_metrics: bool,
) -> VerificationConfig {
    let sstadex_root = manifest.join("../../SSTADEX-prev");
    let pdk_root = env::var_os("PDK_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| sstadex_root.join("IHP-Open-PDK"));
    let ngspice = pdk_root.join("ihp-sg13g2/libs.tech/ngspice");

    VerificationConfig::new(
        manifest.join("examples/ota_4t_electrical_verification/ihp_sg13g2.spice"),
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
    .template_variable("pmos_length", format_float(mirror.length))
    .template_variable("pmos_width", format_float(mirror.total_width))
    .template_variable("pmos_nf", mirror.nf.to_string())
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

pub fn optional_ac_measurements(include_loop_metrics: bool) -> &'static str {
    if include_loop_metrics {
        "meas ac unity_gain_hz when gain_db=0 fall=1\n\
         meas ac phase_at_unity find phase_deg at=unity_gain_hz\n\
         let phase_margin_deg = 180 + phase_at_unity"
    } else {
        "* Shapeic found no unity-gain crossing; UGF and phase margin are omitted."
    }
}

pub fn validate_finger_width(primitive: &str, sizing: SizingSummary) -> Result<(), io::Error> {
    if sizing.finger_width >= IHP_MAX_WIDTH_PER_FINGER {
        return Err(io::Error::other(format!(
            "{primitive} Wf={:.6e} must be less than {:.6e}",
            sizing.finger_width, IHP_MAX_WIDTH_PER_FINGER
        )));
    }
    Ok(())
}

pub fn unique_output_root(
    manifest: &Path,
    directory: &str,
) -> Result<PathBuf, std::time::SystemTimeError> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    Ok(manifest
        .join("../target")
        .join(directory)
        .join(timestamp.to_string()))
}

pub fn format_float(value: f64) -> String {
    format!("{value:.17e}")
}
