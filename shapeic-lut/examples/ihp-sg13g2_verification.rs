use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use shapeic_lut::verification::{VerificationConfig, VerificationEngine, VerificationInput};
use shapeic_lut::{
    DeviceLut, Expr, LookupTable, LutError, MosCapacitanceMatrix, MosExpression,
    MosExtrinsicCapacitances, OperatingPoint,
};

const IHP_MAX_WIDTH_PER_FINGER: f64 = 10.0e-6;

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let sstadex_root = manifest.join("../../SSTADEX-prev");
    let pdk_root = env::var_os("PDK_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| sstadex_root.join("IHP-Open-PDK"));
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let output_root = manifest
        .join("../target/shapeic-lut-verification")
        .join(timestamp.to_string());

    verify_device(
        &sstadex_root.join("LUTs/ihp-sg13g2/lv_5w_nmos.npz"),
        "sg13_lv_nmos",
        manifest.join("examples/verification/ihp-sg13g2/nmos.spice"),
        output_root.join("nmos"),
        &pdk_root,
        OperatingPoint::new(0.4e-6, 0.0, 0.6, 0.6),
        OperatingPoint::new(0.6e-6, -0.05, 0.605, 0.605),
    )?;
    verify_device(
        &sstadex_root.join("LUTs/ihp-sg13g2/lv_5w_pmos.npz"),
        "sg13_lv_pmos",
        manifest.join("examples/verification/ihp-sg13g2/pmos.spice"),
        output_root.join("pmos"),
        &pdk_root,
        OperatingPoint::new(0.4e-6, 0.0, -0.6, -0.6),
        OperatingPoint::new(0.6e-6, 0.05, -0.605, -0.605),
    )?;

    let lengths = [0.4e-6];
    let vgs_values = [0.25];
    let ids_values = [
        10e-6, 150e-6, 200e-6, 300e-6, 380e-6, 400e-6, 500e-6, 1000e-6,
    ];

    let (coupled_points, coupled_ids): (Vec<OperatingPoint>, Vec<f64>) = lengths
        .into_iter()
        .flat_map(|length| {
            vgs_values.into_iter().flat_map(move |vgs| {
                ids_values
                    .into_iter()
                    .map(move |ids| (OperatingPoint::new(length, 0.0, vgs, 0.35), ids))
            })
        })
        .unzip();

    verify_sweep(
        &sstadex_root.join("LUTs/ihp-sg13g2/lv_5w_nmos.npz"),
        "sg13_lv_nmos",
        manifest.join("examples/verification/ihp-sg13g2/nmos.spice"),
        output_root.join("nmos_sweep"),
        &pdk_root,
        coupled_points.clone(),
        coupled_ids.clone(),
    )?;

    verify_sweep_with_wf(
        &sstadex_root.join("LUTs/ihp-sg13g2/lv_5w_nmos.npz"),
        "sg13_lv_nmos",
        manifest.join("examples/verification/ihp-sg13g2/nmos.spice"),
        output_root.join("nmos_sweep_wf"),
        &pdk_root,
        coupled_points.clone(),
        coupled_ids.clone(),
    )?;

    verify_sweep_5d(
        &manifest.join("lut_generation/generated/ihp_sg13g2_lv_nmos_5d.npz"),
        "sg13_lv_nmos",
        manifest.join("examples/verification/ihp-sg13g2/nmos.spice"),
        output_root.join("nmos_sweep_5d"),
        &pdk_root,
        coupled_points,
        coupled_ids,
    )?;

    Ok(())
}

fn verify_device(
    lut_path: &Path,
    model_name: &str,
    template: PathBuf,
    output_dir: PathBuf,
    pdk_root: &Path,
    exact_point: OperatingPoint,
    manipulated_point: OperatingPoint,
) -> Result<(), Box<dyn Error>> {
    let table = LookupTable::open(lut_path)?;
    let model = table.model(model_name)?;
    let lut_width = model.device_parameter("w")?;

    // Everything above VerificationEngine is user-owned calculation. The
    // second input chooses both an off-grid length and a different width.
    let inputs = [
        manually_calculated_input(model, exact_point, lut_width)?,
        manually_calculated_input(model, manipulated_point, 7.5e-6)?,
    ];

    let engine = VerificationEngine::new(ihp_config(template, output_dir, pdk_root))?;
    let report = engine.verify(&inputs)?;
    println!("{model_name}: {}", report.summary_path.display());
    print!("{}", report.render_table());
    Ok(())
}

fn verify_sweep(
    lut_path: &Path,
    model_name: &str,
    template: PathBuf,
    output_dir: PathBuf,
    pdk_root: &Path,
    points: Vec<OperatingPoint>,
    ids: Vec<f64>,
) -> Result<(), Box<dyn Error>> {
    let table = LookupTable::open(lut_path)?;
    let model = table.model(model_name)?;
    let lut_width = model.device_parameter("w")?;

    let mut inputs: Vec<VerificationInput> = Vec::new();
    for (point, id) in points.iter().zip(ids.iter()) {
        inputs.push(manually_calculated_input_v2(model, point, lut_width, id)?);
    }

    let engine = VerificationEngine::new(ihp_config(template, output_dir, pdk_root))?;
    let report = engine.verify(&inputs)?;
    println!("{model_name}: {}", report.summary_path.display());
    print!("{}", report.render_table());
    Ok(())
}

fn verify_sweep_with_wf(
    lut_path: &Path,
    model_name: &str,
    template: PathBuf,
    output_dir: PathBuf,
    pdk_root: &Path,
    points: Vec<OperatingPoint>,
    ids: Vec<f64>,
) -> Result<(), Box<dyn Error>> {
    let table = LookupTable::open(lut_path)?;
    let model = table.model(model_name)?;
    let lut_width = model.device_parameter("w")?;

    let mut inputs: Vec<VerificationInput> = Vec::new();
    for (point, id) in points.iter().zip(ids.iter()) {
        inputs.push(manually_calculated_input_v3(model, point, lut_width, id)?);
    }

    let engine = VerificationEngine::new(ihp_config(template, output_dir, pdk_root))?;
    let report = engine.verify(&inputs)?;
    println!("{model_name}: {}", report.summary_path.display());
    print!("{}", report.render_table());
    Ok(())
}

fn verify_sweep_5d(
    lut_path: &Path,
    model_name: &str,
    template: PathBuf,
    output_dir: PathBuf,
    pdk_root: &Path,
    points: Vec<OperatingPoint>,
    ids: Vec<f64>,
) -> Result<(), Box<dyn Error>> {
    let table = LookupTable::open(lut_path)?;
    let model = table.model(model_name)?;

    let mut inputs: Vec<VerificationInput> = Vec::new();
    for (point, id) in points.iter().zip(ids.iter()) {
        inputs.push(manually_calculated_input_5d(model, point, id)?);
    }

    let engine = VerificationEngine::new(ihp_config(template, output_dir, pdk_root))?;
    let report = engine.verify(&inputs)?;
    println!("{model_name}: {}", report.summary_path.display());
    print!("{}", report.render_table());
    Ok(())
}

fn manually_calculated_input(
    model: &DeviceLut,
    point: OperatingPoint,
    width: f64,
) -> Result<VerificationInput, LutError> {
    let lut_width = model.device_parameter("w")?;
    let width_scale = width / lut_width;

    let id = model.query_parameter(&point, "id")? * width_scale;
    let gm = model.query_parameter(&point, "gm")? * width_scale;
    let gds = model.query_parameter(&point, "gds")? * width_scale;
    let gm_id = gm / id;
    let jd = id / width;

    let nf = minimum_ihp_nf(width);
    Ok(VerificationInput::new(point, width, nf)
        .reference("id", id)
        .reference("gm", gm)
        .reference("gds", gds)
        .reference("gm_id", gm_id)
        .reference("jd", jd))
}

fn manually_calculated_input_v2(
    model: &DeviceLut,
    point: &OperatingPoint,
    lut_width: f64,
    id: &f64,
) -> Result<VerificationInput, LutError> {
    let jd_expr = model.standard_expression(MosExpression::CurrentDensity)?;
    let jd = model.query_expression(point, &jd_expr)?;
    let width = id / jd;
    let width_scale = width / lut_width;
    let lut_id = model.query_parameter(point, "id")?;

    //let gds = model.query_parameter(point, "gds")? * width_scale;
    let gds_id = model.query_parameter(point, "gds")? / lut_id;
    let gds = gds_id * id;

    let gm = model.query_parameter(point, "gm")? * width_scale;
    let gm_id = gm / id;

    let nf = minimum_ihp_nf(width);
    Ok(VerificationInput::new(*point, width, nf)
        .reference("id", *id)
        .reference("gm", gm)
        .reference("gds", gds)
        .reference("gm_id", gm_id)
        .reference("jd", jd))
}

fn manually_calculated_input_v3(
    model: &DeviceLut,
    point: &OperatingPoint,
    lut_width: f64,
    id: &f64,
) -> Result<VerificationInput, LutError> {
    let lut_id = model.query_parameter(point, "id")?;
    let nf = (id / lut_id).round();
    println!("{}", id / lut_id);
    println!("{}", nf);

    let width = lut_width * nf;
    let id_final = lut_id * nf;
    let gds = model.query_parameter(point, "gds")? * nf;
    let gm = model.query_parameter(point, "gm")? * nf;

    let gm_id = gm / id_final;
    let jd = id_final / width;

    //let nf = minimum_ihp_nf(width);
    Ok(VerificationInput::new(*point, width, nf as u32)
        .reference("id", *id)
        .reference("gm", gm)
        .reference("gds", gds)
        .reference("gm_id", gm_id)
        .reference("jd", jd))
}

fn manually_calculated_input_5d(
    model: &DeviceLut,
    point: &OperatingPoint,
    id: &f64,
) -> Result<VerificationInput, LutError> {
    let expressions = ["gds", "gm"]
        .into_iter()
        .chain(MosCapacitanceMatrix::INDEPENDENT_PARAMETERS)
        .map(Expr::parameter)
        .collect::<Vec<_>>();
    let sizing = model.size_for_current(point, *id, &expressions)?;
    println!(
        "Target ID={:.6e}, LUT ID={:.6e}, error={:.6e}, Wf={:.6e}, nf={}",
        sizing.requested_current,
        sizing.predicted_current,
        sizing.current_error,
        sizing.point.finger_width,
        sizing.nf,
    );

    let nf = f64::from(sizing.nf);
    let width = sizing.total_width;
    let id_final = sizing.predicted_current;
    let gds = sizing.values[0] * nf;
    let gm = sizing.values[1] * nf;
    let gm_id = gm / id_final;
    let jd = id_final / width;

    let mut input = VerificationInput::new(*point, width, sizing.nf)
        .reference("id", id_final)
        .reference("gm", gm)
        .reference("gds", gds)
        .reference("gm_id", gm_id)
        .reference("jd", jd);
    let intrinsic =
        MosCapacitanceMatrix::from_flat(&sizing.values[2..])?.scaled(f64::from(sizing.nf));
    for (parameter, value) in MosCapacitanceMatrix::PARAMETERS
        .into_iter()
        .zip(intrinsic.to_ngspice_parameters())
    {
        input = input.reference(parameter, value);
    }
    let extrinsic = sizing.extrinsic_capacitances.ok_or_else(|| {
        LutError::ExtrinsicCapacitanceSamplesUnavailable {
            model: model.name().to_owned(),
        }
    })?;
    for (parameter, value) in MosExtrinsicCapacitances::PARAMETERS.into_iter().zip([
        extrinsic.cgsol,
        extrinsic.cgdol,
        extrinsic.cjs,
        extrinsic.cjd,
    ]) {
        input = input.reference(parameter, value);
    }

    Ok(input)
}

fn ihp_config(template: PathBuf, output_dir: PathBuf, pdk_root: &Path) -> VerificationConfig {
    let ngspice = pdk_root.join("ihp-sg13g2/libs.tech/ngspice");
    VerificationConfig::new(template, output_dir)
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
}

fn minimum_ihp_nf(width: f64) -> u32 {
    ((width / IHP_MAX_WIDTH_PER_FINGER).floor() as u32).saturating_add(1)
}
