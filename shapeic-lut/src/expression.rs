use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Axis, DeviceLut, LutError};

/// A typed arithmetic expression evaluated against LUT values.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Constant(f64),
    Axis(Axis),
    FingerWidth,
    Parameter(String),
    DeviceParameter(String),
    /// A standard MOS expression carrying its canonical output name.
    Mos {
        /// Semantic identity used to name the expression result.
        kind: MosExpression,
        /// Arithmetic expression evaluated against the LUT.
        expression: Box<Expr>,
    },
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

impl Expr {
    pub const fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    pub const fn axis(axis: Axis) -> Self {
        Self::Axis(axis)
    }

    pub const fn finger_width() -> Self {
        Self::FingerWidth
    }

    pub fn parameter(name: impl Into<String>) -> Self {
        Self::Parameter(name.into())
    }

    /// Returns the output name of a direct LUT parameter or standard MOS expression.
    pub fn parameter_name(&self) -> Option<&str> {
        match self {
            Self::Parameter(name) => Some(name),
            Self::Mos { kind, .. } => Some(kind.parameter_name()),
            _ => None,
        }
    }

    pub fn device_parameter(name: impl Into<String>) -> Self {
        Self::DeviceParameter(name.into())
    }

    pub(crate) fn evaluate_at(&self, model: &DeviceLut, index: &[usize]) -> Result<f64, LutError> {
        let value = match self {
            Self::Constant(value) => *value,
            Self::Axis(axis) => model.axis(*axis)[index[axis.index()]],
            Self::FingerWidth => model.finger_width_at(index)?,
            Self::Parameter(name) => {
                if let Some(array) = model.parameters.get(name) {
                    array.scalar_or_grid_value(index).ok_or_else(|| {
                        LutError::schema(
                            format!("model '{}'.parameter '{name}'", model.name()),
                            format!("index {index:?} is outside shape {:?}", array.shape()),
                        )
                    })?
                } else {
                    model
                        .parameter_expression(name)?
                        .evaluate_at(model, index)?
                }
            }
            Self::DeviceParameter(name) => model.device_parameter(name)?,
            Self::Mos { expression, .. } => expression.evaluate_at(model, index)?,
            Self::Neg(expression) => -expression.evaluate_at(model, index)?,
            Self::Add(left, right) => {
                left.evaluate_at(model, index)? + right.evaluate_at(model, index)?
            }
            Self::Sub(left, right) => {
                left.evaluate_at(model, index)? - right.evaluate_at(model, index)?
            }
            Self::Mul(left, right) => {
                left.evaluate_at(model, index)? * right.evaluate_at(model, index)?
            }
            Self::Div(left, right) => {
                left.evaluate_at(model, index)? / right.evaluate_at(model, index)?
            }
        };

        if value.is_finite() {
            Ok(value)
        } else {
            Err(LutError::NonFinite {
                model: model.name().to_owned(),
                context: format!("expression '{self}'"),
            })
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(value) => write!(formatter, "{value}"),
            Self::Axis(axis) => formatter.write_str(axis.as_str()),
            Self::FingerWidth => formatter.write_str("finger_width"),
            Self::Parameter(name) => formatter.write_str(name),
            Self::DeviceParameter(name) => write!(formatter, "device.{name}"),
            Self::Mos { expression, .. } => expression.fmt(formatter),
            Self::Neg(expression) => write!(formatter, "(-{expression})"),
            Self::Add(left, right) => write!(formatter, "({left} + {right})"),
            Self::Sub(left, right) => write!(formatter, "({left} - {right})"),
            Self::Mul(left, right) => write!(formatter, "({left} * {right})"),
            Self::Div(left, right) => write!(formatter, "({left} / {right})"),
        }
    }
}

macro_rules! impl_binary_operator {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl $trait for Expr {
            type Output = Expr;

            fn $method(self, rhs: Expr) -> Self::Output {
                Expr::$variant(Box::new(self), Box::new(rhs))
            }
        }

        impl $trait<f64> for Expr {
            type Output = Expr;

            fn $method(self, rhs: f64) -> Self::Output {
                self.$method(Expr::constant(rhs))
            }
        }

        impl $trait<Expr> for f64 {
            type Output = Expr;

            fn $method(self, rhs: Expr) -> Self::Output {
                Expr::constant(self).$method(rhs)
            }
        }
    };
}

impl_binary_operator!(Add, add, Add);
impl_binary_operator!(Sub, sub, Sub);
impl_binary_operator!(Mul, mul, Mul);
impl_binary_operator!(Div, div, Div);

impl Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Self::Output {
        Self::Neg(Box::new(self))
    }
}

/// Standard MOS expressions corresponding to the useful non-plotting gmid expressions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MosExpression {
    Vsg,
    Vsb,
    Vsd,
    Gmid,
    Vov,
    Vstar,
    CurrentDensity,
    IntrinsicGain,
    TransitFrequency,
    EarlyVoltage,
    InverseEarlyVoltage,
    Rds,
    Vdsat,
}

impl MosExpression {
    /// Returns the canonical analog-design name for this expression.
    pub const fn parameter_name(self) -> &'static str {
        match self {
            Self::Vsg => "vsg",
            Self::Vsb => "vsb",
            Self::Vsd => "vsd",
            Self::Gmid => "gmid",
            Self::Vov => "vov",
            Self::Vstar => "vstar",
            Self::CurrentDensity => "jd",
            Self::IntrinsicGain => "av",
            Self::TransitFrequency => "ft",
            Self::EarlyVoltage => "va",
            Self::InverseEarlyVoltage => "gdsid",
            Self::Rds => "rds",
            Self::Vdsat => "vdsat",
        }
    }
}

pub(crate) fn standard_expression(
    model: &DeviceLut,
    expression: MosExpression,
) -> Result<Expr, LutError> {
    let parameter = |name| model.parameter_expression(name);

    let inner = match expression {
        MosExpression::Vsg => -Expr::axis(Axis::Vgs),
        MosExpression::Vsb => -Expr::axis(Axis::Vbs),
        MosExpression::Vsd => -Expr::axis(Axis::Vds),
        MosExpression::Gmid => parameter("gm")? / parameter("id")?,
        MosExpression::Vov => Expr::axis(Axis::Vgs) - parameter("vth")?,
        MosExpression::Vstar => 2.0 * parameter("id")? / parameter("gm")?,
        MosExpression::CurrentDensity => parameter("id")? / width_expression(model)?,
        MosExpression::IntrinsicGain => parameter("gm")? / parameter("gds")?,
        MosExpression::TransitFrequency => {
            parameter("gm")? / (2.0 * std::f64::consts::PI * parameter("cgg")?)
        }
        MosExpression::EarlyVoltage => parameter("id")? / parameter("gds")?,
        MosExpression::InverseEarlyVoltage => parameter("gds")? / parameter("id")?,
        MosExpression::Rds => 1.0 / parameter("gds")?,
        MosExpression::Vdsat => {
            let name = ["vdssat", "vdsat", "vsat"]
                .into_iter()
                .find(|name| model.parameters.contains_key(*name))
                .ok_or_else(|| LutError::UnknownParameter {
                    model: model.name().to_owned(),
                    parameter: "vdssat/vdsat/vsat".to_owned(),
                })?;
            parameter(name)?
        }
    };

    Ok(Expr::Mos {
        kind: expression,
        expression: Box::new(inner),
    })
}

fn width_expression(model: &DeviceLut) -> Result<Expr, LutError> {
    if model.finger_widths().is_some() {
        return Ok(Expr::finger_width());
    }

    if let Some(width) = model.device_parameters.get("w") {
        return Ok(Expr::constant(*width));
    }

    let effective_width = model.parameter_expression("weff")?;
    match model.device_parameters.get("nf") {
        Some(fingers) => Ok(effective_width * *fingers),
        None => Ok(effective_width),
    }
}
