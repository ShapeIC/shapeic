use std::io::Cursor;
use std::path::PathBuf;

use shapeic_lut::{
    Axis, DType, Expr, LookupTable, LutError, LutPoint, MosExpression, OperatingPoint,
};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sstadex_combined.npz")
}

fn fixture() -> LookupTable {
    LookupTable::open(fixture_path()).expect("fixture should load")
}

fn five_dimensional_fixture() -> LookupTable {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/shapeic_v2_5d.npz");
    LookupTable::open(path).expect("five-dimensional fixture should load")
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
        .standard_expression(MosExpression::Gmid)
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
fn standard_mos_expressions_expose_canonical_parameter_names() {
    let table = fixture();
    let model = table.model("fixture_nmos").expect("nmos");
    let cases = [
        (MosExpression::Vsg, "vsg"),
        (MosExpression::Vsb, "vsb"),
        (MosExpression::Vsd, "vsd"),
        (MosExpression::Gmid, "gmid"),
        (MosExpression::Vov, "vov"),
        (MosExpression::Vstar, "vstar"),
        (MosExpression::CurrentDensity, "jd"),
        (MosExpression::IntrinsicGain, "av"),
        (MosExpression::TransitFrequency, "ft"),
        (MosExpression::EarlyVoltage, "va"),
        (MosExpression::InverseEarlyVoltage, "gdsid"),
        (MosExpression::Rds, "rds"),
        (MosExpression::Vdsat, "vdsat"),
    ];

    for (kind, expected_name) in cases {
        assert_eq!(kind.parameter_name(), expected_name);
        assert_eq!(
            model
                .standard_expression(kind)
                .expect("standard expression")
                .parameter_name(),
            Some(expected_name),
        );
    }

    let manual_gmid = Expr::parameter("gm") / Expr::parameter("id");
    assert_eq!(manual_gmid.parameter_name(), None);
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

#[test]
fn loads_shapeic_v2_metadata_and_five_dimensional_arrays() {
    let table = five_dimensional_fixture();
    assert_eq!(table.description(), Some("IHP SG13G2 LV NMOS smoke LUT"));
    assert_eq!(table.simulator(), Some("ngspice"));
    assert_eq!(table.model_names().collect::<Vec<_>>(), ["sg13_lv_nmos"]);

    let model = table.model("sg13_lv_nmos").expect("nmos");
    assert_eq!(model.axis(Axis::Length), [0.4e-6, 0.8e-6]);
    assert_eq!(model.axis(Axis::Vbs), [0.0, -0.1]);
    assert_eq!(model.finger_widths(), Some([0.5e-6, 1.0e-6].as_slice()));
    assert_eq!(model.array("id").expect("id").dtype(), DType::F32);
    assert_eq!(model.array("id").expect("id").shape(), [2, 2, 2, 2, 2]);
    assert_eq!(model.device_parameter("nf").expect("nf"), 1.0);
}

#[test]
fn interpolates_five_dimensions_and_expressions_on_corners() {
    let table = five_dimensional_fixture();
    let model = table.model("sg13_lv_nmos").expect("nmos");
    let midpoint = LutPoint::new(OperatingPoint::new(0.6e-6, -0.05, 0.5, 0.5), 0.75e-6);
    let id = model.array("id").expect("id");
    let gm = model.array("gm").expect("gm");
    let mut id_sum = 0.0;
    let mut gmid_sum = 0.0;
    let mut jd_sum = 0.0;
    for corner in 0..32 {
        let index = (0..5)
            .map(|dimension| usize::from(corner & (1 << dimension) != 0))
            .collect::<Vec<_>>();
        let corner_id = id.get_f64(&index).expect("id corner");
        id_sum += corner_id;
        gmid_sum += gm.get_f64(&index).expect("gm corner") / corner_id;
        jd_sum += corner_id / model.finger_widths().expect("widths")[index[4]];
    }

    assert_close(
        model
            .query_parameter_at(&midpoint, "id")
            .expect("five-dimensional id"),
        id_sum / 32.0,
    );
    let gmid = model
        .standard_expression(MosExpression::Gmid)
        .expect("gmid");
    assert_close(
        model
            .query_expression_at(&midpoint, &gmid)
            .expect("five-dimensional gmid"),
        gmid_sum / 32.0,
    );
    let jd = model
        .standard_expression(MosExpression::CurrentDensity)
        .expect("current density");
    assert_close(
        model
            .query_expression_at(&midpoint, &jd)
            .expect("five-dimensional current density"),
        jd_sum / 32.0,
    );
}

#[test]
fn keeps_four_and_five_dimensional_query_apis_explicit() {
    let five_dimensional = five_dimensional_fixture();
    let model = five_dimensional.model("sg13_lv_nmos").expect("nmos");
    let operating_point = OperatingPoint::new(0.4e-6, 0.0, 0.4, 0.4);
    assert!(matches!(
        model.query_parameter(&operating_point, "id"),
        Err(LutError::FingerWidthRequired { .. })
    ));
    assert!(matches!(
        model.query_parameter_at(&LutPoint::new(operating_point, 2.0e-6), "id"),
        Err(LutError::FingerWidthOutOfRange { .. })
    ));
    assert!(matches!(
        model.query_parameter_at(&LutPoint::new(operating_point, f64::NAN), "id"),
        Err(LutError::NonFinite { .. })
    ));

    let legacy = fixture();
    let model = legacy.model("fixture_nmos").expect("legacy nmos");
    assert!(matches!(
        model.query_parameter_at(&LutPoint::new(operating_point, 2.0), "id"),
        Err(LutError::NoFingerWidthAxis { .. })
    ));
}

#[test]
fn evaluates_five_dimensional_batches() {
    let table = five_dimensional_fixture();
    let model = table.model("sg13_lv_nmos").expect("nmos");
    let points = [
        LutPoint::new(OperatingPoint::new(0.4e-6, 0.0, 0.4, 0.4), 0.5e-6),
        LutPoint::new(OperatingPoint::new(0.8e-6, -0.1, 0.6, 0.6), 1.0e-6),
    ];
    let values = model
        .query_many_at(&points, &[Expr::parameter("id"), Expr::finger_width()])
        .expect("five-dimensional batch");
    assert_eq!(values.len(), 2);
    assert_close(values[0][1], 0.5e-6);
    assert_close(values[1][1], 1.0e-6);
}

#[test]
fn sizes_for_current_and_returns_per_finger_expressions() {
    let table = five_dimensional_fixture();
    let model = table.model("sg13_lv_nmos").expect("nmos");
    let operating_point = OperatingPoint::new(0.4e-6, 0.0, 0.4, 0.4);
    let widths = model.finger_widths().expect("finger widths");
    let minimum_current = model
        .query_parameter_at(&LutPoint::new(operating_point, widths[0]), "id")
        .expect("minimum-width current");
    let maximum_current = model
        .query_parameter_at(
            &LutPoint::new(operating_point, widths[widths.len() - 1]),
            "id",
        )
        .expect("maximum-width current");
    let requested_current = (minimum_current + maximum_current) / 2.0;
    let gm = Expr::parameter("gm");
    let gmid = model
        .standard_expression(MosExpression::Gmid)
        .expect("gmid expression");

    let result = model
        .size_for_current(
            &operating_point,
            requested_current,
            &[gm.clone(), gmid.clone()],
        )
        .expect("inverse current sizing");

    assert_eq!(result.nf, 1);
    assert_close(result.point.finger_width, (widths[0] + widths[1]) / 2.0);
    assert_close(result.total_width, result.point.finger_width);
    assert_close(result.predicted_current, requested_current);
    assert_close(result.current_error, 0.0);
    assert_close(
        result.values[0],
        model
            .query_expression_at(&result.point, &gm)
            .expect("per-finger gm"),
    );
    assert_eq!(gmid.parameter_name(), Some("gmid"));
    assert_close(
        result.values[1],
        model
            .query_expression_at(&result.point, &gmid)
            .expect("per-finger gmid"),
    );
}
