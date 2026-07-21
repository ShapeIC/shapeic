use shapeic_lut::{Axis, Expr, LookupTable, OperatingPoint};
use std::time::Instant;

fn main() {
    let start = Instant::now();
    let table = LookupTable::open("../SSTADEX-prev/LUTs/ihp-sg13g2/lv_5w_nmos.npz")
        .expect("could open ihp-sg13g2 LUT");
    let elapsed = start.elapsed();
    let models = table.model_names().collect::<Vec<_>>();
    let nmos_model = models[0];

    println!("LookupTable::open took: {elapsed:?}");
    println!("{:?}", table.description());
    println!("{:?}", table.simulator());
    println!("{:?}", models);

    let nmos = table.model(nmos_model).expect("model not found");
    let nmos_vgs = nmos.axis(Axis::Vgs);

    println!("{:?}", nmos_vgs);

    let query = nmos.query_parameter(&OperatingPoint::new(0.4e-6, 0.0, 1.0, 1.0), "id");
    println!("{:?}", query);

    let lengths = [0.4e-6, 0.8e-6];
    let vgs_values = [0.1, 0.2, 0.3, 0.4, 0.5];

    let coupled_points: Vec<OperatingPoint> = lengths
        .iter()
        .flat_map(|&length| {
            vgs_values
                .iter()
                .map(move |&vgs| OperatingPoint::new(length, 0.0, vgs, 0.6))
        })
        .collect();

    let paired_points: Vec<_> = lengths
        .iter()
        .zip(vgs_values.iter())
        .map(|(&length, &vgs)| OperatingPoint::new(length, 0.0, vgs, 0.6))
        .collect();

    let id_many = nmos.query_many(&coupled_points, &[Expr::parameter("id")]);
    println!("{:?}", id_many);
    let id_paired = nmos.query_many(&paired_points, &[Expr::parameter("id")]);
    println!("{:?}", id_paired);
}
