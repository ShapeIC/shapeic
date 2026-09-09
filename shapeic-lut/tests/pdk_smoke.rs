use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use shapeic_lut::{
    Axis, DType, DeviceLut, LookupTable, LutPoint, MosCapacitanceMatrix, MosExpression,
    MosExtrinsicCapacitances, OperatingPoint,
};

const OPEN_PDKS_REVISION: &str = "026824c7969ce6f4fc9678e6ca04b0a06a596c4b";
const PARAMETER_TOLERANCE: f64 = 1.0e-5;
const MATRIX_ERROR_LIMIT: f64 = 0.01;

struct FixtureSpec {
    prefix: &'static str,
    pdk: &'static str,
    revision: &'static str,
    corner: &'static str,
    temperature_c: f64,
    nominal_voltage: f64,
    lengths: [f64; 2],
    finger_widths: [f64; 2],
    vgs_magnitudes: [f64; 2],
    vds_magnitudes: [f64; 2],
}

const SKY130: FixtureSpec = FixtureSpec {
    prefix: "sky130a_1v8",
    pdk: "sky130A",
    revision: OPEN_PDKS_REVISION,
    corner: "tt",
    temperature_c: 27.0,
    nominal_voltage: 1.8,
    lengths: [0.15e-6, 0.3e-6],
    finger_widths: [0.42e-6, 0.92e-6],
    vgs_magnitudes: [0.6, 0.8],
    vds_magnitudes: [0.6, 0.8],
};

const GF180: FixtureSpec = FixtureSpec {
    prefix: "gf180mcud_3v3",
    pdk: "gf180mcuD",
    revision: OPEN_PDKS_REVISION,
    corner: "typical",
    temperature_c: 25.0,
    nominal_voltage: 3.3,
    lengths: [0.28e-6, 0.56e-6],
    finger_widths: [0.22e-6, 0.72e-6],
    vgs_magnitudes: [0.8, 1.2],
    vds_magnitudes: [0.8, 1.2],
};

#[derive(Debug, Deserialize)]
struct GoldenReference {
    format: String,
    version: u32,
    device: String,
    pdk: String,
    pdk_revision: String,
    corner: String,
    temperature_c: f64,
    point: GoldenPoint,
    frequencies_hz: Vec<f64>,
    nf_results: Vec<GoldenNfResult>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct GoldenPoint {
    length: f64,
    finger_width: f64,
    vbs: f64,
    vgs: f64,
    vds: f64,
}

#[derive(Debug, Deserialize)]
struct GoldenNfResult {
    nf: u32,
    canonical_parameters: BTreeMap<String, f64>,
    ac_capacitance_f: Vec<Vec<Vec<f64>>>,
}

struct SmokeFixture {
    spec: &'static FixtureSpec,
    table: LookupTable,
    reference: GoldenReference,
}

fn fixture(spec: &'static FixtureSpec, name: &str) -> SmokeFixture {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let table = LookupTable::open(root.join(format!("{}_{name}_smoke.npz", spec.prefix)))
        .expect("electrical smoke fixture should load");
    let reference = read_reference(&root.join(format!("{}_{name}_reference.json", spec.prefix)));
    SmokeFixture {
        spec,
        table,
        reference,
    }
}

fn read_reference(path: &Path) -> GoldenReference {
    serde_json::from_slice(&fs::read(path).expect("golden reference should open"))
        .expect("golden reference should parse")
}

fn exact_point(reference: &GoldenReference) -> LutPoint {
    let point = reference.point;
    LutPoint::new(
        OperatingPoint::new(point.length, point.vbs, point.vgs, point.vds),
        point.finger_width,
    )
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64, context: &str) {
    let error = (actual - expected).abs() / expected.abs().max(1.0e-30);
    assert!(
        error <= tolerance,
        "{context}: expected {expected:.16e}, got {actual:.16e}, relative error {error:.6e}"
    );
}

fn matrix_relative_error(actual: &[[f64; 4]; 4], expected: &[Vec<f64>]) -> f64 {
    let mut difference_squared = 0.0;
    let mut expected_squared = 0.0;
    for row in 0..4 {
        for column in 0..4 {
            difference_squared += (actual[row][column] - expected[row][column]).powi(2);
            expected_squared += expected[row][column].powi(2);
        }
    }
    difference_squared.sqrt() / expected_squared.sqrt().max(1.0e-18)
}

fn combined_matrix(
    intrinsic: MosCapacitanceMatrix,
    extrinsic: MosExtrinsicCapacitances,
) -> [[f64; 4]; 4] {
    let mut matrix = intrinsic.values;
    for (first, second, value) in [
        (0, 2, extrinsic.cgsol),
        (0, 1, extrinsic.cgdol),
        (2, 3, extrinsic.cjs),
        (1, 3, extrinsic.cjd),
    ] {
        matrix[first][first] += value;
        matrix[second][second] += value;
        matrix[first][second] -= value;
        matrix[second][first] -= value;
    }
    matrix
}

fn charge_residual(matrix: &[[f64; 4]; 4]) -> f64 {
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0e-30);
    let maximum = (0..4).fold(0.0_f64, |maximum, index| {
        let row = matrix[index].iter().sum::<f64>().abs();
        let column = matrix.iter().map(|values| values[index]).sum::<f64>().abs();
        maximum.max(row).max(column)
    });
    maximum / scale
}

fn verify_metadata(fixture: &SmokeFixture) {
    let spec = fixture.spec;
    let reference = &fixture.reference;
    assert_eq!(reference.format, "shapeic-electrical-reference");
    assert_eq!(reference.version, 1);
    assert_eq!(reference.pdk, spec.pdk);
    assert_eq!(reference.pdk_revision, spec.revision);
    assert_eq!(reference.corner, spec.corner);
    assert_eq!(reference.temperature_c, spec.temperature_c);
    assert_eq!(reference.frequencies_hz, [1.0e6, 1.0e7]);
    assert_eq!(fixture.table.pdk(), Some(reference.pdk.as_str()));
    assert_eq!(
        fixture.table.pdk_revision(),
        Some(reference.pdk_revision.as_str())
    );
    assert_eq!(fixture.table.corner(), Some(reference.corner.as_str()));
    assert_eq!(fixture.table.temperature_c(), Some(reference.temperature_c));
    assert_eq!(fixture.table.nominal_voltage(), Some(spec.nominal_voltage));
    assert_eq!(
        fixture.table.model_names().collect::<Vec<_>>(),
        [reference.device.as_str()]
    );
    assert_eq!(
        reference
            .nf_results
            .iter()
            .map(|result| result.nf)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );

    let model = fixture
        .table
        .model(&reference.device)
        .expect("referenced smoke model should exist");
    assert_eq!(model.axis(Axis::Length), spec.lengths);
    assert_eq!(model.finger_widths(), Some(spec.finger_widths.as_slice()));
    let sign = if reference.device.contains("nfet") {
        1.0
    } else {
        -1.0
    };
    assert_eq!(model.axis(Axis::Vbs), [0.0, -sign * 0.1]);
    assert_eq!(
        model.axis(Axis::Vgs),
        spec.vgs_magnitudes.map(|value| sign * value)
    );
    assert_eq!(
        model.axis(Axis::Vds),
        spec.vds_magnitudes.map(|value| sign * value)
    );
    assert_eq!(model.array("id").expect("id array").dtype(), DType::F32);
    assert_eq!(
        model.array("id").expect("id array").shape(),
        [2, 2, 2, 2, 2]
    );
    for parameter in [
        "weff",
        "id",
        "vth",
        "vdsat",
        "gm",
        "gds",
        "cgg",
        "cgd",
        "cgs",
        "cdg",
        "cdd",
        "cds",
        "csg",
        "csd",
        "css",
        "cgsol",
        "cgdol",
        "cjs",
        "cjd",
        "cgsol_nf4",
        "cgdol_nf4",
        "cjs_nf4",
        "cjd_nf4",
    ] {
        assert!(
            model.parameter_names().iter().any(|name| name == parameter),
            "missing canonical smoke-LUT parameter {parameter}"
        );
    }
}

fn verify_exact_corner(fixture: &SmokeFixture) {
    let reference = &fixture.reference;
    let model = fixture
        .table
        .model(&reference.device)
        .expect("referenced smoke model should exist");
    let point = exact_point(reference);
    let golden = reference
        .nf_results
        .iter()
        .find(|result| result.nf == 1)
        .expect("golden reference should contain nf=1");

    for parameter in ["id", "gm", "gds", "weff", "vth", "vdsat"] {
        let expected = golden.canonical_parameters[parameter];
        let actual = model
            .query_parameter_at(&point, parameter)
            .expect("exact parameter query should succeed");
        assert_relative(actual, expected, PARAMETER_TOLERANCE, parameter);
    }

    let parameters = &golden.canonical_parameters;
    let expression_cases = [
        (MosExpression::Gmid, parameters["gm"] / parameters["id"]),
        (
            MosExpression::InverseEarlyVoltage,
            parameters["gds"] / parameters["id"],
        ),
        (
            MosExpression::CurrentDensity,
            parameters["id"] / point.finger_width,
        ),
        (
            MosExpression::IntrinsicGain,
            parameters["gm"] / parameters["gds"],
        ),
        (
            MosExpression::TransitFrequency,
            parameters["gm"] / (2.0 * std::f64::consts::PI * parameters["cgg"]),
        ),
    ];
    for (kind, expected) in expression_cases {
        let expression = model
            .standard_expression(kind)
            .expect("standard expression should exist");
        let actual = model
            .query_expression_at(&point, &expression)
            .expect("exact expression query should succeed");
        assert_relative(
            actual,
            expected,
            PARAMETER_TOLERANCE * 2.0,
            kind.parameter_name(),
        );
    }

    let intrinsic = model
        .query_capacitance_matrix_at(&point)
        .expect("intrinsic capacitance query should succeed");
    let extrinsic = model
        .query_extrinsic_capacitances_at(&point, 1)
        .expect("extrinsic capacitance query should succeed");
    let combined = combined_matrix(intrinsic, extrinsic);
    assert!(charge_residual(&combined) <= 1.0e-5);
    for matrix in &golden.ac_capacitance_f {
        assert!(matrix_relative_error(&combined, matrix) <= MATRIX_ERROR_LIMIT);
    }
}

fn midpoint(model: &DeviceLut) -> LutPoint {
    let middle = |values: &[f64]| (values[0] + values[values.len() - 1]) / 2.0;
    LutPoint::new(
        OperatingPoint::new(
            middle(model.axis(Axis::Length)),
            middle(model.axis(Axis::Vbs)),
            middle(model.axis(Axis::Vgs)),
            middle(model.axis(Axis::Vds)),
        ),
        middle(model.finger_widths().expect("smoke LUT should have widths")),
    )
}

fn corners(model: &DeviceLut) -> Vec<LutPoint> {
    let widths = model.finger_widths().expect("smoke LUT should have widths");
    (0..32)
        .map(|corner| {
            let select = |dimension: usize, values: &[f64]| {
                values[usize::from(corner & (1 << dimension) != 0)]
            };
            LutPoint::new(
                OperatingPoint::new(
                    select(0, model.axis(Axis::Length)),
                    select(1, model.axis(Axis::Vbs)),
                    select(2, model.axis(Axis::Vgs)),
                    select(3, model.axis(Axis::Vds)),
                ),
                select(4, widths),
            )
        })
        .collect()
}

fn verify_midpoint_interpolation(fixture: &SmokeFixture) {
    let model = fixture
        .table
        .model(&fixture.reference.device)
        .expect("referenced smoke model should exist");
    let midpoint = midpoint(model);
    let corners = corners(model);

    for parameter in [
        "id", "gm", "gds", "cgg", "cgd", "cgs", "cdg", "cdd", "cds", "csg", "csd", "css", "cgsol",
        "cgdol", "cjs", "cjd",
    ] {
        let expected = corners
            .iter()
            .map(|point| {
                model
                    .query_parameter_at(point, parameter)
                    .expect("corner parameter should query")
            })
            .sum::<f64>()
            / corners.len() as f64;
        let actual = model
            .query_parameter_at(&midpoint, parameter)
            .expect("midpoint parameter should query");
        assert_relative(actual, expected, 1.0e-12, parameter);
    }

    for kind in [
        MosExpression::Gmid,
        MosExpression::InverseEarlyVoltage,
        MosExpression::CurrentDensity,
        MosExpression::IntrinsicGain,
        MosExpression::TransitFrequency,
    ] {
        let expression = model
            .standard_expression(kind)
            .expect("standard expression should exist");
        let expected = corners
            .iter()
            .map(|point| {
                model
                    .query_expression_at(point, &expression)
                    .expect("corner expression should query")
            })
            .sum::<f64>()
            / corners.len() as f64;
        let actual = model
            .query_expression_at(&midpoint, &expression)
            .expect("midpoint expression should query");
        assert_relative(actual, expected, 1.0e-12, kind.parameter_name());
    }

    let intrinsic = model
        .query_capacitance_matrix_at(&midpoint)
        .expect("midpoint capacitance should query");
    let extrinsic = model
        .query_extrinsic_capacitances_at(&midpoint, 3)
        .expect("midpoint extrinsic capacitance should query");
    assert!(charge_residual(&combined_matrix(intrinsic, extrinsic)) <= 1.0e-5);
}

fn verify_sizing_and_extrinsic_scaling(fixture: &SmokeFixture) {
    let model = fixture
        .table
        .model(&fixture.reference.device)
        .expect("referenced smoke model should exist");
    let golden_point = exact_point(&fixture.reference).operating_point;
    let operating_point = OperatingPoint::new(
        model.axis(Axis::Length)[0],
        golden_point.vbs,
        golden_point.vgs,
        golden_point.vds,
    );
    let widths = model.finger_widths().expect("smoke LUT should have widths");
    let narrowest_point = LutPoint::new(operating_point, widths[0]);
    let widest_point = LutPoint::new(operating_point, widths[widths.len() - 1]);
    let minimum_current = model
        .query_parameter_at(&narrowest_point, "id")
        .expect("minimum-width current should query");
    let maximum_current = model
        .query_parameter_at(&widest_point, "id")
        .expect("maximum-width current should query");
    let requested_current = minimum_current + maximum_current;
    let gmid = model
        .standard_expression(MosExpression::Gmid)
        .expect("gmid should exist");
    let result = model
        .size_for_current(&operating_point, requested_current, &[gmid])
        .expect("smoke-LUT sizing should succeed");

    assert_eq!(result.nf, 2);
    assert!(result.point.finger_width >= widths[0]);
    assert!(result.point.finger_width <= widths[widths.len() - 1]);
    assert_relative(
        result.total_width,
        result.point.finger_width * f64::from(result.nf),
        1.0e-12,
        "total width",
    );
    assert_relative(
        result.predicted_current,
        requested_current,
        1.0e-6,
        "predicted current",
    );
    assert_eq!(result.values.len(), 1);
    assert!(result.extrinsic_capacitances.is_some());

    let base = model
        .query_extrinsic_capacitances_at(&widest_point, 1)
        .expect("nf=1 extrinsics should query");
    for nf in 1..=6 {
        let actual = model
            .query_extrinsic_capacitances_at(&widest_point, nf)
            .expect("linearly extrapolated extrinsics should query");
        let factor = f64::from(nf);
        for (name, actual, expected) in [
            ("cgsol", actual.cgsol, base.cgsol * factor),
            ("cgdol", actual.cgdol, base.cgdol * factor),
            ("cjs", actual.cjs, base.cjs * factor),
            ("cjd", actual.cjd, base.cjd * factor),
        ] {
            assert_relative(actual, expected, PARAMETER_TOLERANCE, name);
        }
    }
}

#[test]
fn validates_sky130_nmos_smoke_lut() {
    let fixture = fixture(&SKY130, "nmos");
    verify_metadata(&fixture);
    verify_exact_corner(&fixture);
    verify_midpoint_interpolation(&fixture);
    verify_sizing_and_extrinsic_scaling(&fixture);
}

#[test]
fn validates_sky130_pmos_smoke_lut() {
    let fixture = fixture(&SKY130, "pmos");
    verify_metadata(&fixture);
    verify_exact_corner(&fixture);
    verify_midpoint_interpolation(&fixture);
    verify_sizing_and_extrinsic_scaling(&fixture);
}

#[test]
fn validates_gf180_nmos_smoke_lut() {
    let fixture = fixture(&GF180, "nmos");
    assert_eq!(fixture.reference.point.length, GF180.lengths[0]);
    assert_eq!(fixture.reference.point.finger_width, GF180.finger_widths[0]);
    verify_metadata(&fixture);
    verify_exact_corner(&fixture);
    verify_midpoint_interpolation(&fixture);
    verify_sizing_and_extrinsic_scaling(&fixture);
}

#[test]
fn validates_gf180_pmos_smoke_lut() {
    let fixture = fixture(&GF180, "pmos");
    assert_eq!(fixture.reference.point.length, GF180.lengths[0]);
    assert_eq!(fixture.reference.point.finger_width, GF180.finger_widths[0]);
    verify_metadata(&fixture);
    verify_exact_corner(&fixture);
    verify_midpoint_interpolation(&fixture);
    verify_sizing_and_extrinsic_scaling(&fixture);
}
