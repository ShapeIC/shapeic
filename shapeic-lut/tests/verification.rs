use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use shapeic_lut::verification::{
    MetricStatus, VerificationConfig, VerificationEngine, VerificationInput, VerificationStatus,
};
use shapeic_lut::{DeviceLut, LookupTable, LutError, OperatingPoint};

fn unique_temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "shapeic-lut-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ))
}

#[cfg(unix)]
#[test]
fn engine_compares_manual_values_and_continues_after_ngspice_failure() {
    use std::os::unix::fs::PermissionsExt;

    let root = unique_temp_dir("fake-engine");
    fs::create_dir(&root).expect("temp root");
    let template = root.join("template.spice");
    fs::write(
        &template,
        concat!(
            "WIDTH {{width}}\n",
            "LENGTH {{length}}\n",
            "VBS {{vbs}}\n",
            "VGS {{vgs}}\n",
            "VDS {{vds}}\n",
            "NF {{nf}}\n",
            "PMOS_WIDTH {{pmos_width}}\n",
            "RESULTS {{results_path}}\n",
        ),
    )
    .expect("template");
    let simulator = root.join("fake-ngspice.sh");
    fs::write(
        &simulator,
        concat!(
            "#!/bin/sh\n",
            "log=$4\n",
            "netlist=$5\n",
            "printf 'fake ngspice\\n' > \"$log\"\n",
            "case \"$netlist\" in *point_000002*) exit 7 ;; esac\n",
            "results=$(sed -n 's/^RESULTS //p' \"$netlist\")\n",
            "printf 'scale id gm gds gm_id jd\\n0 8.5 16 2 1.8823529411764706 4.25\\n' > \"$results\"\n",
        ),
    )
    .expect("fake simulator");
    let mut permissions = fs::metadata(&simulator).expect("metadata").permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&simulator, permissions).expect("permissions");

    let output = root.join("output");
    let mut config =
        VerificationConfig::new(&template, &output).dynamic_template_variable("pmos_width");
    config.ngspice = simulator;
    config.max_width_per_finger = Some(10.0);
    let engine = VerificationEngine::new(config).expect("engine");
    let point = OperatingPoint::new(2.0, 0.0, 1.0, 2.0);
    let input = VerificationInput::new(point, 2.0, 1)
        .reference("id", 8.5)
        .reference("gm", 16.0)
        .reference("gds", 2.0)
        .reference("gm_id", 4.0)
        .reference("jd", 4.25)
        .dynamic_template_variable("pmos_width", "5e-6");
    let width_override = VerificationInput::new(point, 3.0, 7)
        .reference("id", 8.5)
        .reference("gm", 16.0)
        .reference("gds", 2.0)
        .reference("gm_id", 1.8823529411764706)
        .dynamic_template_variable("pmos_width", "7e-6");
    let failed = VerificationInput::new(OperatingPoint::new(1.0, -1.0, 0.0, 0.5), 4.0, 2)
        .reference("id", 1.0)
        .reference("gm", 2.0)
        .reference("gds", 3.0)
        .reference("gm_id", 2.0)
        .dynamic_template_variable("pmos_width", "9e-6");
    let inputs = [input, width_override, failed];
    let report = engine.verify(&inputs).expect("report");

    assert_eq!(report.runs.len(), 3);
    assert_eq!(report.runs[0].status, VerificationStatus::Success);
    assert_eq!(report.runs[1].status, VerificationStatus::Success);
    assert_eq!(report.runs[2].status, VerificationStatus::Failed);
    assert_eq!(report.runs[1].input.width, 3.0);
    assert_eq!(report.runs[1].input.nf, 7);
    assert_eq!(report.rows.len(), 13);
    let id = report
        .rows
        .iter()
        .find(|row| row.point_index == 0 && row.metric == "id")
        .expect("id row");
    assert_eq!(id.reference_value, 8.5);
    assert_eq!(id.absolute_error, Some(0.0));
    let gmid = report
        .rows
        .iter()
        .find(|row| row.point_index == 0 && row.metric == "gm_id")
        .expect("gm_id row");
    assert!(gmid.percent_error.expect("percentage") > 100.0);
    let jd = report
        .rows
        .iter()
        .find(|row| row.point_index == 0 && row.metric == "jd")
        .expect("jd row");
    assert_eq!(jd.simulation_value, Some(4.25));
    assert_eq!(jd.absolute_error, Some(0.0));

    assert!(report.summary_path.is_file());
    assert!(report.runs[0].artifact_dir.join("netlist.spice").is_file());
    assert!(report.runs[0].artifact_dir.join("ngspice.log").is_file());
    assert!(report.runs[0].artifact_dir.join("results.tsv").is_file());
    let overridden_netlist = fs::read_to_string(report.runs[1].artifact_dir.join("netlist.spice"))
        .expect("rendered netlist");
    assert!(overridden_netlist.contains("WIDTH 3.00000000000000000e0"));
    assert!(overridden_netlist.contains("LENGTH 2.00000000000000000e0"));
    assert!(overridden_netlist.contains("NF 7"));
    assert!(overridden_netlist.contains("PMOS_WIDTH 7e-6"));
    assert!(report.runs[2].artifact_dir.join("netlist.spice").is_file());
    assert!(report.runs[2].artifact_dir.join("ngspice.log").is_file());
    assert!(!report.runs[2].artifact_dir.join("results.tsv").exists());

    let mut csv = csv::Reader::from_path(&report.summary_path).expect("summary CSV");
    let headers = csv.headers().expect("CSV headers").clone();
    assert!(headers.iter().any(|header| header == "reference_value"));
    assert!(headers.iter().any(|header| header == "simulation_value"));
    assert!(headers.iter().any(|header| header == "nf"));
    assert_eq!(csv.records().count(), 13);

    let table = report.render_table();
    assert!(table.contains("LUT/manual"));
    assert!(table.contains("| nf "));
    assert!(table.contains("8.500000e0"));
    assert!(table.contains("1.600000e1"));
    assert!(engine.verify(&inputs).is_err());

    let mut single_config = engine.config().clone();
    single_config.output_dir = root.join("single-output");
    let single_input = VerificationInput::new(point, 2.0, 1)
        .reference("id", 8.5)
        .reference("vth", 0.3)
        .dynamic_template_variable("pmos_width", "5e-6");
    let single_report = VerificationEngine::new(single_config)
        .expect("single engine")
        .verify_one(&single_input)
        .expect("single report");
    assert_eq!(single_report.runs[0].status, VerificationStatus::Partial);
    assert_eq!(single_report.rows[0].status, MetricStatus::Compared);
    assert_eq!(single_report.rows[1].status, MetricStatus::SimulationError);
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn rejects_invalid_manual_inputs_before_creating_output() {
    let root = unique_temp_dir("invalid-input");
    fs::create_dir(&root).expect("temp root");
    let template = root.join("template.spice");
    fs::write(
        &template,
        "{{width}} {{length}} {{vbs}} {{vgs}} {{vds}} {{results_path}}",
    )
    .expect("template");
    let output = root.join("output");
    let engine = VerificationEngine::new(
        VerificationConfig::new(&template, &output).max_width_per_finger(10.0e-6),
    )
    .expect("engine");

    let no_references =
        VerificationInput::new(OperatingPoint::new(1.0e-6, 0.0, 1.0, 1.0), 1.0e-6, 1);
    assert!(engine.verify_one(&no_references).is_err());
    let invalid_length = VerificationInput::new(OperatingPoint::new(0.0, 0.0, 1.0, 1.0), 1.0e-6, 1)
        .reference("id", 1.0);
    assert!(engine.verify_one(&invalid_length).is_err());
    let invalid_value =
        VerificationInput::new(OperatingPoint::new(1.0e-6, 0.0, 1.0, 1.0), 1.0e-6, 1)
            .reference("id", f64::NAN);
    assert!(engine.verify_one(&invalid_value).is_err());
    let zero_fingers =
        VerificationInput::new(OperatingPoint::new(1.0e-6, 0.0, 1.0, 1.0), 1.0e-6, 0)
            .reference("id", 1.0);
    assert!(engine.verify_one(&zero_fingers).is_err());
    let width_at_limit =
        VerificationInput::new(OperatingPoint::new(1.0e-6, 0.0, 1.0, 1.0), 20.0e-6, 2)
            .reference("id", 1.0);
    assert!(engine.verify_one(&width_at_limit).is_err());
    assert!(!output.exists());

    let invalid_config =
        VerificationConfig::new(&template, root.join("invalid-config")).max_width_per_finger(0.0);
    assert!(VerificationEngine::new(invalid_config).is_err());

    let dynamic = VerificationEngine::new(
        VerificationConfig::new(&template, root.join("dynamic-output"))
            .dynamic_template_variable("pmos_width"),
    )
    .expect("dynamic engine");
    let missing_dynamic =
        VerificationInput::new(OperatingPoint::new(1.0e-6, 0.0, 1.0, 1.0), 1.0e-6, 1)
            .reference("id", 1.0);
    assert!(dynamic.verify_one(&missing_dynamic).is_err());
    let unexpected_dynamic = missing_dynamic
        .clone()
        .dynamic_template_variable("unexpected", "1");
    assert!(dynamic.verify_one(&unexpected_dynamic).is_err());

    let conflict = VerificationConfig::new(&template, root.join("conflict"))
        .template_variable("pmos_width", "1")
        .dynamic_template_variable("pmos_width");
    assert!(VerificationEngine::new(conflict).is_err());
    let invalid_name = VerificationConfig::new(&template, root.join("invalid-name"))
        .dynamic_template_variable("pmos-width");
    assert!(VerificationEngine::new(invalid_name).is_err());
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
#[ignore = "requires NGSpice and a local IHP SG13G2 PDK"]
fn verifies_real_ihp_nmos_manual_values() {
    verify_real_ihp_values(
        "nmos",
        "lv_5w_nmos.npz",
        "sg13_lv_nmos",
        OperatingPoint::new(0.6e-6, -0.05, 0.605, 0.605),
    );
}

#[test]
#[ignore = "requires NGSpice and a local IHP SG13G2 PDK"]
fn verifies_real_ihp_pmos_manual_values() {
    verify_real_ihp_values(
        "pmos",
        "lv_5w_pmos.npz",
        "sg13_lv_pmos",
        OperatingPoint::new(0.6e-6, 0.05, -0.605, -0.605),
    );
}

fn verify_real_ihp_values(
    device: &str,
    lut_file: &str,
    model_name: &str,
    operating_point: OperatingPoint,
) {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let sstadex = std::env::var_os("SSTADEX_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../SSTADEX-prev"));
    let pdk_root = std::env::var_os("PDK_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| sstadex.join("IHP-Open-PDK"));
    let ngspice = pdk_root.join("ihp-sg13g2/libs.tech/ngspice");
    let output = unique_temp_dir("ihp-integration");
    let config = VerificationConfig::new(
        manifest.join(format!("examples/verification/ihp-sg13g2/{device}.spice")),
        &output,
    )
    .max_width_per_finger(10.0e-6)
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
    );

    // LUT access and width scaling happen outside VerificationEngine.
    let table = LookupTable::open(sstadex.join("LUTs/ihp-sg13g2").join(lut_file)).expect("IHP LUT");
    let model = table.model(model_name).expect("IHP model");
    let input = manual_reference(model, operating_point, 7.5e-6).expect("manual reference");
    let report = VerificationEngine::new(config)
        .expect("engine")
        .verify_one(&input)
        .expect("verification");
    assert_eq!(report.runs[0].status, VerificationStatus::Success);
    assert!(report.rows.iter().all(|row| row.percent_error.is_some()));
    fs::remove_dir_all(output).expect("cleanup");
}

fn manual_reference(
    model: &DeviceLut,
    point: OperatingPoint,
    width: f64,
) -> Result<VerificationInput, LutError> {
    let width_scale = width / model.device_parameter("w")?;
    let id = model.query_parameter(&point, "id")? * width_scale;
    let gm = model.query_parameter(&point, "gm")? * width_scale;
    let gds = model.query_parameter(&point, "gds")? * width_scale;
    let jd = id / width;
    Ok(VerificationInput::new(point, width, 1)
        .reference("id", id)
        .reference("gm", gm)
        .reference("gds", gds)
        .reference("gm_id", gm / id)
        .reference("jd", jd))
}
