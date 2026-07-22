use std::error::Error;
use std::path::Path;

use shapeic_lut::{LookupTable, LutPoint, OperatingPoint};

fn main() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("lut_generation/generated");
    let cases = [
        (
            "ihp_sg13g2_lv_nmos_5d.npz",
            "sg13_lv_nmos",
            LutPoint::new(OperatingPoint::new(0.4e-6, 0.0, 0.4, 0.4), 0.5e-6),
        ),
        (
            "ihp_sg13g2_lv_pmos_smoke.npz",
            "sg13_lv_pmos",
            LutPoint::new(OperatingPoint::new(0.4e-6, 0.0, -0.4, -0.4), 0.5e-6),
        ),
    ];

    for (filename, model_name, point) in cases {
        let table = LookupTable::open(root.join(filename))?;
        let model = table.model(model_name)?;
        let id = model.query_parameter_at(&point, "id")?;
        println!(
            "{model_name}: Id={id:.6e} A at Wf={:.6e} m",
            point.finger_width
        );
    }
    Ok(())
}
