#![warn(missing_docs)]
//! Modified nodal analysis utilities for ShapeIC.
//!
//! This crate provides symbolic and numerical MNA construction, SPICE
//! preprocessing, circuit evaluation, and utilities for inspecting generated
//! MNA systems.

/// Utilities for formatting and inspecting MNA systems and their solutions.
pub mod evaluation;
/// High-level modified nodal analysis construction and symbolic solution.
pub mod mna;
/// Numerical evaluation and AC solution of prepared MNA systems.
pub mod numeric;
/// SPICE netlist preprocessing and node mapping.
pub mod spice2cir;
/// Symbolic modified nodal analysis construction from SPICE-like netlists.
pub mod symmna;
