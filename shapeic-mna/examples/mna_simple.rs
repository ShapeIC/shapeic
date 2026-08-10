use std::path::Path;

use shapeic_mna::mna::{mna, mna_solve};

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let spice_dir = manifest.join("examples/spice") ;
    let output_dir = manifest.join("examples/outputs");

    let system = mna(
        &spice_dir,
        &output_dir,
        "mna_simple"
    ).unwrap();

    let solution = mna_solve(
        &system.a,
        &system.x,
        &system.z
    ).unwrap();

    println!("{}", system.report);
    println!("{}", solution.solution);
}
