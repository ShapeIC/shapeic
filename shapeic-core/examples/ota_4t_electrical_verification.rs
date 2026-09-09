#[allow(dead_code)]
#[path = "ota_4t_electrical_verification/analysis.rs"]
mod analysis;
#[path = "ota_4t_electrical_verification/common.rs"]
mod common;

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};

use analysis::{AcMetrics, ElectricalAnalysis, SizingSummary};
use common::{
    NMOS_MODEL, PMOS_MODEL, TAIL_CURRENT, VDD_DC, VG_DC, VOUT_DC, unique_output_root,
    validate_finger_width, verification_config,
};
use shapeic_lut::verification::{VerificationEngine, VerificationInput};
use shapeic_lut::{LookupTable, OperatingPoint};

const DIFF_LENGTH: f64 = 0.8e-6;
const MIRROR_LENGTH: f64 = 1.6e-6;
const SOURCE_VOLTAGE: f64 = 0.7122;

fn main() -> Result<(), Box<dyn Error>> {
    let (nmos_path, pmos_path) = lut_paths()?;
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output_root = unique_output_root(manifest, "shapeic-ota-4t-electrical-verification")?;
    let diff_point = OperatingPoint::new(
        DIFF_LENGTH,
        0.0,
        VG_DC - SOURCE_VOLTAGE,
        VOUT_DC - SOURCE_VOLTAGE,
    );
    let mirror_point = OperatingPoint::new(MIRROR_LENGTH, 0.0, VOUT_DC - VDD_DC, VOUT_DC - VDD_DC);

    let nmos_table = LookupTable::open(nmos_path)?;
    let pmos_table = LookupTable::open(pmos_path)?;
    let nmos = nmos_table.model(NMOS_MODEL)?;
    let pmos = pmos_table.model(PMOS_MODEL)?;
    let electrical = analysis::analyze(
        nmos,
        pmos,
        diff_point,
        mirror_point,
        TAIL_CURRENT / 2.0,
        &manifest.join("examples/ota_4t"),
        &output_root.join("mna"),
    )?;
    validate_finger_width("simplediffpair", electrical.diff_pair)?;
    validate_finger_width("currentmirror", electrical.current_mirror)?;
    let references = comparison_metrics(electrical.metrics)?;

    print_sizing(&electrical, diff_point, mirror_point);
    let mut input = VerificationInput::new(
        diff_point,
        electrical.diff_pair.total_width,
        electrical.diff_pair.nf,
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
        electrical.current_mirror,
        references.has_loop_metrics(),
    ))?;
    let report = engine.verify_one(&input)?;
    let run = report
        .runs
        .first()
        .ok_or_else(|| io::Error::other("verification produced no run artifacts"))?;
    let shapeic_ac_path = run.artifact_dir.join("shapeic_ac.csv");
    let ngspice_ac_path = run.artifact_dir.join("ngspice_ac.tsv");
    electrical.ac_sweep.write_csv(&shapeic_ac_path)?;
    let plot_helper = manifest.join("examples/ota_4t_electrical_verification/plot_ac.py");

    println!();
    println!("Shapeic electrical MNA vs. transistor-level ngspice");
    print!("{}", report.render_table());
    println!("CSV: {}", report.summary_path.display());
    println!(
        "Rendered NGSpice netlist: {}",
        run.artifact_dir.join("netlist.spice").display()
    );
    println!(
        "NGSpice operating-point log: {}",
        run.artifact_dir.join("ngspice.log").display()
    );
    println!("Shapeic AC sweep: {}", shapeic_ac_path.display());
    println!("NGSpice AC sweep: {}", ngspice_ac_path.display());
    println!(
        "AC comparison plot: python3 {} {}",
        plot_helper.display(),
        run.artifact_dir.display()
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
            "Shapeic electrical AC did not find {metric} inside {:.6e}..{:.6e} Hz",
            analysis::AC_MIN_HZ,
            analysis::AC_MAX_HZ
        ))
    };
    if metrics.unity_gain_hz.is_some() != metrics.phase_margin_deg.is_some() {
        return Err(io::Error::other(
            "Shapeic electrical AC returned only one of unity-gain frequency and phase margin",
        ));
    }
    Ok(ComparisonMetrics {
        dc_gain_db: metrics.dc_gain_db.ok_or_else(|| missing("DC gain"))?,
        bandwidth_3db_hz: metrics
            .bandwidth_3db_hz
            .ok_or_else(|| missing("the -3 dB bandwidth"))?,
        unity_gain_hz: metrics.unity_gain_hz,
        phase_margin_deg: metrics.phase_margin_deg,
    })
}

fn print_sizing(
    electrical: &ElectricalAnalysis,
    diff_point: OperatingPoint,
    mirror_point: OperatingPoint,
) {
    println!("Electrical OTA verification point");
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
    print_sizing_row("simplediffpair", electrical.diff_pair, diff_point);
    print_sizing_row("currentmirror", electrical.current_mirror, mirror_point);
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

fn lut_paths() -> Result<(PathBuf, PathBuf), io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| OsString::from("ota_4t_electrical_verification"));
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "usage: {} <nmos-5d.npz> <pmos-5d.npz>",
                Path::new(&executable).display()
            ),
        )
    };
    let nmos = arguments.next().ok_or_else(&usage)?;
    let pmos = arguments.next().ok_or_else(&usage)?;
    if arguments.next().is_some() {
        return Err(usage());
    }
    Ok((nmos.into(), pmos.into()))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn configured_operating_points_match_the_ota_example() {
        let diff = OperatingPoint::new(
            DIFF_LENGTH,
            0.0,
            VG_DC - SOURCE_VOLTAGE,
            VOUT_DC - SOURCE_VOLTAGE,
        );
        let mirror = OperatingPoint::new(MIRROR_LENGTH, 0.0, VOUT_DC - VDD_DC, VOUT_DC - VDD_DC);
        assert!((diff.vgs - 0.25).abs() < 1.0e-12);
        assert!((diff.vds - 0.35).abs() < 1.0e-12);
        assert!((mirror.vgs + 0.5).abs() < 1.0e-12);
        assert!((mirror.vds + 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn accepts_optional_unity_gain_metrics_as_a_pair() {
        let complete = AcMetrics {
            dc_gain_db: Some(20.0),
            bandwidth_3db_hz: Some(1.0e6),
            unity_gain_hz: Some(10.0e6),
            phase_margin_deg: Some(60.0),
        };
        let complete = comparison_metrics(complete).expect("complete metrics");
        assert_eq!(complete.loop_metrics(), Some((10.0e6, 60.0)));

        let without_crossing = AcMetrics {
            dc_gain_db: Some(20.0),
            bandwidth_3db_hz: Some(1.0e6),
            unity_gain_hz: None,
            phase_margin_deg: None,
        };
        assert!(
            comparison_metrics(without_crossing)
                .expect("gain and bandwidth suffice")
                .loop_metrics()
                .is_none()
        );

        let inconsistent = AcMetrics {
            phase_margin_deg: Some(60.0),
            ..without_crossing
        };
        assert!(comparison_metrics(inconsistent).is_err());
    }

    #[test]
    fn verification_template_is_fully_bound() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let template = fs::read_to_string(
            manifest.join("examples/ota_4t_electrical_verification/ihp_sg13g2.spice"),
        )
        .expect("verification template");
        assert!(template.contains("let loop_response = -v(VOUT)"));
        assert!(template.contains("let phase_deg = 180 / pi * cph(loop_response)"));
        assert!(common::optional_ac_measurements(true).contains("180 + phase_at_unity"));

        let mirror = SizingSummary {
            length: MIRROR_LENGTH,
            finger_width: 1.0e-6,
            nf: 2,
            total_width: 2.0e-6,
            requested_current: 10.0e-6,
            predicted_current: 10.0e-6,
            current_error: 0.0,
        };
        let config = verification_config(manifest, Path::new("unused-output"), mirror, false);
        VerificationEngine::new(config).expect("valid verification template");
        let config = verification_config(manifest, Path::new("unused-output"), mirror, true);
        VerificationEngine::new(config).expect("valid verification template with loop metrics");
    }

    #[test]
    #[ignore = "requires ngspice and a local IHP SG13G2 PDK"]
    fn real_template_exports_four_scalar_metrics_and_total_widths() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let output_root = std::env::temp_dir().join(format!("shapeic-ota-4t-template-{timestamp}"));
        let mirror = SizingSummary {
            length: MIRROR_LENGTH,
            finger_width: 9.0e-6,
            nf: 2,
            total_width: 18.0e-6,
            requested_current: 10.0e-6,
            predicted_current: 10.0e-6,
            current_error: 0.0,
        };
        let diff_width = 19.063498e-6;
        let diff_point = OperatingPoint::new(DIFF_LENGTH, 0.0, 0.25, 0.35);
        let input = VerificationInput::new(diff_point, diff_width, 2)
            .reference("dc_gain_db", 1.0)
            .reference("bandwidth_3db_hz", 1.0)
            .reference("unity_gain_hz", 1.0)
            .reference("phase_margin_deg", 1.0);
        let report =
            VerificationEngine::new(verification_config(manifest, &output_root, mirror, true))
                .expect("verification engine")
                .verify_one(&input)
                .expect("ngspice verification");

        assert_eq!(report.rows.len(), 4);
        assert!(report.rows.iter().all(|row| row.simulation_value.is_some()));
        let netlist =
            fs::read_to_string(report.runs[0].artifact_dir.join("netlist.spice")).expect("netlist");
        assert!(netlist.contains(&format!("w={} ng=2", common::format_float(diff_width))));
        assert!(netlist.contains(&format!(
            "w={} ng=2",
            common::format_float(mirror.total_width)
        )));
        assert!(netlist.contains("XDP1 VOUT VINP IBIAS IBIAS"));
        assert_ngspice_ac_sweep(&report.runs[0].artifact_dir.join("ngspice_ac.tsv"));
        fs::remove_dir_all(output_root).expect("cleanup");
    }

    #[test]
    #[ignore = "requires ngspice and a local IHP SG13G2 PDK"]
    fn real_template_omits_unavailable_unity_gain_metrics() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let output_root =
            std::env::temp_dir().join(format!("shapeic-ota-4t-no-ugf-template-{timestamp}"));
        let mirror = SizingSummary {
            length: MIRROR_LENGTH,
            finger_width: 9.0e-6,
            nf: 2,
            total_width: 18.0e-6,
            requested_current: 10.0e-6,
            predicted_current: 10.0e-6,
            current_error: 0.0,
        };
        let diff_point = OperatingPoint::new(DIFF_LENGTH, 0.0, 0.25, 0.35);
        let input = VerificationInput::new(diff_point, 19.063498e-6, 2)
            .reference("dc_gain_db", 1.0)
            .reference("bandwidth_3db_hz", 1.0);
        let report =
            VerificationEngine::new(verification_config(manifest, &output_root, mirror, false))
                .expect("verification engine")
                .verify_one(&input)
                .expect("ngspice verification");

        assert_eq!(report.rows.len(), 2);
        assert!(report.rows.iter().all(|row| row.simulation_value.is_some()));
        let netlist =
            fs::read_to_string(report.runs[0].artifact_dir.join("netlist.spice")).expect("netlist");
        assert!(!netlist.contains("meas ac unity_gain_hz"));
        assert_ngspice_ac_sweep(&report.runs[0].artifact_dir.join("ngspice_ac.tsv"));
        fs::remove_dir_all(output_root).expect("cleanup");
    }

    fn assert_ngspice_ac_sweep(path: &Path) {
        let text = fs::read_to_string(path).expect("NGSpice AC sweep");
        let mut lines = text.lines().filter(|line| !line.trim().is_empty());
        let headers = lines
            .next()
            .expect("AC sweep header")
            .split_whitespace()
            .collect::<Vec<_>>();
        assert_eq!(
            headers,
            [
                "frequency",
                "response_real",
                "response_imag",
                "gain_db",
                "phase_deg"
            ]
        );
        let mut previous_frequency = 0.0;
        let mut row_count = 0;
        for line in lines {
            let values = line
                .split_whitespace()
                .map(|value| value.parse::<f64>().expect("finite numeric AC value"))
                .collect::<Vec<_>>();
            assert_eq!(values.len(), headers.len());
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(values[0] > previous_frequency);
            previous_frequency = values[0];
            row_count += 1;
        }
        assert_eq!(row_count, 221);
    }
}
