use std::env;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::ffi::OsString;
use std::error::Error;
use std::time::Instant;

use shapeic_lut::LookupTable;
use shapeic_core::utils::{linspace};
use shapeic_core::catalog::primitive_loader::load_primitive_catalog;
use shapeic_core::primitive::build::{PrimitiveBuildInput, PrimitiveBuildValue, build_candidate_set_for_primitive, PrimitiveBuildError};
use shapeic_mna::mna::{mna};
use shapeic_mna::numeric::PreparedNumericMna;
    

const TAIL_CURRENT: f64 = 20.0e-6;
const LENGTHS: [f64; 5] = [0.4e-6, 0.8e-6, 1.6e-6, 3.2e-6, 6.4e-6];
const VOUT: f64 = 1.0;
const VDD: f64 = 1.5;
const VIN: f64 = 0.9;
const VTAIL_START: f64 = 0.65;
const VTAIL_STOP: f64 = 0.79;
const VTAIL_POINTS: usize = 10;
const NMOS_MODEL: &str = "sg13_lv_nmos";
const PMOS_MODEL: &str = "sg13_lv_pmos";
const PHYSICAL_LAYOUT_POLICY: &str = "symmetric-adjacent-with-edge-dummies-v3";
const AC_MIN_HZ: f64 = 1.0;
const AC_MAX_HZ: f64 = 100.0e9;
#[cfg(test)]
const AC_POINTS_PER_DECADE: usize = 20;
const AC_COARSE_POINTS_PER_DECADE: usize = 4;
const AC_CROSSING_RELATIVE_TOLERANCE: f64 = 0.005;
const AC_MAX_REFINEMENT_STEPS: usize = 32;
const DC_GAIN_PARAMETER_ORDER: [&str; 4] = ["g_gm_xdp", "r_gds_xdp", "g_gm_xcm", "r_gds_xcm"];
//const ANALYSIS_MODE: AnalysisMode = AnalysisMode::Prune;
//const ANALYSIS_TARGETS: AnalysisTargets = AnalysisTargets {
//    min_dc_gain_db: Some(25.0),
//    min_bandwidth_3db_hz: Some(100.0e4),
//    min_unity_gain_hz: Some(1.0e7),
//    min_phase_margin_deg: Some(45.0),
//};

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let spice_dir = manifest.join("examples/ota_4t");
    let output_dir = manifest.join("../target/shapeic-ota-4t-v2");
    let primitives_dir = manifest.join("../analoglib/primitives/");
    let (nmos_path, pmos_path, physical_path) = lut_paths()?;
    
    let stage_start = Instant::now();
    let nmos_table = LookupTable::open(nmos_path)?;
    let pmos_table = LookupTable::open(pmos_path)?;
    let lut_load = stage_start.elapsed();
    println!("The LUT load took: {:?}", lut_load);
    let nmos = nmos_table.model(NMOS_MODEL)?;
    let pmos = pmos_table.model(PMOS_MODEL)?;

    let catalog = load_primitive_catalog(&primitives_dir).map_err(|error| format!("{error:?}"))?;
    for primitive in &catalog.list() {
        println!("name: {:?}, description: {:?}", primitive.name, primitive.description);
    }

    let branch_current = TAIL_CURRENT/2.0;
    
    // MNA
    let base_system = mna(
        &spice_dir,
        &output_dir,
        "ota_4t"
    ).unwrap(); //TODO: fix this.

    let mut electrical_mna = PreparedNumericMna::new(
        &base_system, 
        &DC_GAIN_PARAMETER_ORDER
    ).map_err(|error| {
        io::Error::other(format!("could not prepare numerical OTA AC MNA: {error}"))
    })?;

    let diffpair = catalog.get("simplediffpair")
        .ok_or_else(|| "missing simplediffpair primitive".to_string())?;

    let diffpair_input = PrimitiveBuildInput::new(HashMap::from([
        ("current".to_string(), PrimitiveBuildValue::Scalar(TAIL_CURRENT)),
        ("VINP".to_string(), PrimitiveBuildValue::Scalar(VIN)),
        ("VOUTP".to_string(), PrimitiveBuildValue::Scalar(VOUT)),
        ("VTAIL".to_string(), PrimitiveBuildValue::Vector(linspace(VTAIL_START, VTAIL_STOP, VTAIL_POINTS))),
    ]));

    let stage_start = Instant::now();
    let diffpair_candidate_set = build_candidate_set_for_primitive(
        nmos,
        diffpair, 
        "xdp",
        diffpair_input
    ).map_err(|error| format!("{error:?}"))?;
    let diffpair_candidate_set_time = stage_start.elapsed();
    println!("Diffpair candidate set took: {:?}", diffpair_candidate_set_time);

    println!("diffpair_candidate_set: {:?}", diffpair_candidate_set);

    let total = total_start.elapsed();
    println!("Total time: {:?}", total);
    Ok(())
}

fn lut_paths() -> Result<(PathBuf, PathBuf, Option<PathBuf>), io::Error> {
    let mut arguments = env::args_os();
    let executable = arguments.next().unwrap_or_else(|| OsString::from("ota_4t"));
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "usage: {} <nmos-5d.npz> <pmos-5d.npz> [physical.npz]",
                Path::new(&executable).display()
            ),
        )
    };
    let nmos = arguments.next().ok_or_else(&usage)?;
    let pmos = arguments.next().ok_or_else(&usage)?;
    let physical = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        return Err(usage());
    }
    Ok((nmos.into(), pmos.into(), physical))
}

