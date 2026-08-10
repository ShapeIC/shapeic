use std::path::Path;
use std::collections::BTreeMap;
use ndarray::array;

use shapeic_mna::mna::{mna, mna_solve};
use shapeic_mna::numeric::{PreparedNumericMna};

fn main() {let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let spice_dir = manifest.join("examples/spice") ;
    let output_dir = manifest.join("examples/outputs");

    let system = mna(
        &spice_dir,
        &output_dir,
        "r_divider"
    ).unwrap();

    let solution = mna_solve(
        &system.a,
        &system.x,
        &system.z
    ).unwrap();

    println!("{:?}", system.x);
    println!("{}", solution.solution);
    println!("{:?}", system.nodes);

    let mut electrical_mna = PreparedNumericMna::new(
        &system,
        &["r"]
    ).unwrap();

    let mut electrical_system = electrical_mna.instantiate(&[100.0]).unwrap();

    let ports = vec!["P".to_owned(), "N".to_owned()];
    let connections = BTreeMap::from([
        ("P".to_owned(), "VOUT".to_owned()),
        ("N".to_owned(), "VSS".to_owned())
    ]);

    electrical_system.stamp_port_admittance(
        &ports, 
        &connections,
        array![[10.0, -10.0], [10.0, -10.0]].view(),
        array![[2.0, -2.0], [-2.0, 2.0]].view(),
    ).unwrap();

    let v1 = electrical_system.solve_node(0.0, "VOUT").unwrap().unwrap();

    println!("{}", v1);

}
