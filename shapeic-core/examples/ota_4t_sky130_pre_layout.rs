#[path = "ota_4t_pre_layout/common.rs"]
mod common;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    common::run(common::sky130_spec())
}
