use std::io::Cursor;
use std::path::PathBuf;

use shapeic_lut::{Axis, DType, Expr, LookupTable, LutError, MosExpression, OperatingPoint};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sstadex_combined.npz")
}

fn fixture() -> LookupTable {
    LookupTable::open(fixture_path()).expect("fixture should load")
}

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-6 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.16e}, got {actual:.16e}"
    );
}

#[test]
fn loads_sstadex_metadata_models_and_typed_arrays() {
    let table = fixture();
    assert_eq!(table.description(), Some("shapeic-lut test fixture"));
    assert_eq!(table.simulator(), Some("FixtureSimulator"));
    assert_eq!(
        table.model_names().collect::<Vec<_>>(),
        vec!["fixture_nmos", "fixture_pmos"]
    );

    for name in ["fixture_nmos", "fixture_pmos"] {
        let model = table.model(name).expect("model should exist");
        assert_eq!(model.parameter_names().len(), 10);
        assert_eq!(model.array("id").expect("id").dtype(), DType::F32);
        assert_eq!(model.array("id").expect("id").shape(), [2, 2, 2, 2]);
        assert_eq!(model.array("vth").expect("vth").shape(), [1]);
        assert_eq!(model.device_parameter("w").expect("width"), 2.0);
    }

    let nmos = table.model("fixture_nmos").expect("nmos");
    assert_eq!(nmos.axis(Axis::Length), [1.0, 3.0]);
    assert_eq!(nmos.axis(Axis::Vgs), [0.0, 2.0]);
    assert_eq!(
        nmos.array("id").expect("id").get_f64(&[1, 1, 1, 1]),
        Some(16.0)
    );

    let pmos = table.model("fixture_pmos").expect("pmos");
    assert_eq!(pmos.axis(Axis::Length), [3.0, 1.0]);
    assert_eq!(pmos.axis(Axis::Vds), [0.0, -4.0]);
}

#[test]
fn interpolates_exact_corners_and_four_dimensional_midpoints() {
    let table = fixture();
    let nmos = table.model("fixture_nmos").expect("nmos");
    assert_close(
        nmos.query_parameter(&OperatingPoint::new(1.0, -1.0, 0.0, 0.0), "id")
            .expect("corner query"),
        1.0,
    );
    assert_close(
        nmos.query_parameter(&OperatingPoint::new(3.0, 1.0, 2.0, 4.0), "id")
            .expect("corner query"),
        16.0,
    );
    assert_close(
        nmos.query_parameter(&OperatingPoint::new(2.0, 0.0, 1.0, 2.0), "id")
            .expect("midpoint query"),
        8.5,
    );

    let pmos = table.model("fixture_pmos").expect("pmos");
    assert_close(
        pmos.query_parameter(&OperatingPoint::new(2.0, 0.0, -1.0, -2.0), "id")
            .expect("descending midpoint query"),
        8.5,
    );
}

#[test]
fn evaluates_expressions_on_corners_before_interpolation() {
    let table = fixture();
    let model = table.model("fixture_nmos").expect("nmos");
    let point = OperatingPoint::new(2.0, 0.0, 1.0, 2.0);
    let gmid = model
        .standard_expression(MosExpression::GmOverId)
        .expect("gmid expression");
    let expected = (1..=16).map(|id| 16.0 / f64::from(id)).sum::<f64>() / 16.0;
    let actual = model.query_expression(&point, &gmid).expect("gmid query");
    assert_close(actual, expected);
    assert!((actual - 16.0 / 8.5).abs() > 0.1);
}

#[test]
fn evaluates_standard_mos_expressions() {
    let table = fixture();
    let model = table.model("fixture_nmos").expect("nmos");
    let point = OperatingPoint::new(2.0, 0.0, 1.0, 2.0);

    let cases = [
        (MosExpression::Vsg, -1.0),
        (MosExpression::Vsb, 0.0),
        (MosExpression::Vsd, -2.0),
        (MosExpression::Vov, 0.75),
        (MosExpression::Vstar, 8.5 / 8.0),
        (MosExpression::CurrentDensity, 4.25),
        (MosExpression::IntrinsicGain, 8.0),
        (MosExpression::TransitFrequency, 2.0 / std::f64::consts::PI),
        (MosExpression::EarlyVoltage, 4.25),
        (MosExpression::Rds, 0.5),
        (MosExpression::Vdsat, 0.2),
    ];
    for (kind, expected) in cases {
        let expression = model
            .standard_expression(kind)
            .expect("standard expression");
        assert_close(
            model
                .query_expression(&point, &expression)
                .expect("standard expression query"),
            expected,
        );
    }

    let expected_inverse_early = (1..=16).map(|id| 2.0 / f64::from(id)).sum::<f64>() / 16.0;
    let inverse_early = model
        .standard_expression(MosExpression::InverseEarlyVoltage)
        .expect("inverse Early expression");
    assert_close(
        model
            .query_expression(&point, &inverse_early)
            .expect("inverse Early query"),
        expected_inverse_early,
    );
}

#[test]
fn evaluates_custom_expressions_and_point_major_batches() {
    let table = fixture();
    let model = table.model("fixture_nmos").expect("nmos");
    let expression = (Expr::parameter("gm") + 2.0) / Expr::parameter("gds");
    let points = [
        OperatingPoint::new(1.0, -1.0, 0.0, 0.0),
        OperatingPoint::new(2.0, 0.0, 1.0, 2.0),
    ];
    let outputs = model
        .query_many(&points, &[Expr::parameter("id"), expression])
        .expect("batch query");
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0], [1.0, 9.0]);
    assert_eq!(outputs[1], [8.5, 9.0]);
}

#[test]
fn reports_query_and_format_errors() {
    let table = fixture();
    let model = table.model("fixture_nmos").expect("nmos");

    assert!(matches!(
        model.query_parameter(&OperatingPoint::new(0.0, 0.0, 1.0, 2.0), "id"),
        Err(LutError::OutOfRange {
            axis: Axis::Length,
            ..
        })
    ));
    assert!(matches!(
        model.query_parameter(&OperatingPoint::new(2.0, 0.0, f64::NAN, 2.0), "id"),
        Err(LutError::NonFinite { .. })
    ));
    assert!(matches!(
        model.query_parameter(&OperatingPoint::new(2.0, 0.0, 1.0, 2.0), "missing"),
        Err(LutError::UnknownParameter { .. })
    ));

    let division_by_zero = Expr::constant(1.0) / 0.0;
    assert!(matches!(
        model.query_expression(&OperatingPoint::new(2.0, 0.0, 1.0, 2.0), &division_by_zero),
        Err(LutError::NonFinite { .. })
    ));
    assert!(matches!(
        model.query_many(
            &[OperatingPoint::new(2.0, 0.0, 1.0, 2.0)],
            &[division_by_zero]
        ),
        Err(LutError::Batch {
            point_index: 0,
            expression_index: 0,
            ..
        })
    ));

    assert!(matches!(
        LookupTable::from_reader(Cursor::new(b"not a zip archive")),
        Err(LutError::Zip(_))
    ));

    let empty_zip = zip::ZipWriter::new(Cursor::new(Vec::new()))
        .finish()
        .expect("empty ZIP should be valid");
    assert!(matches!(
        LookupTable::from_reader(empty_zip),
        Err(LutError::Zip(zip::result::ZipError::FileNotFound))
    ));
}
