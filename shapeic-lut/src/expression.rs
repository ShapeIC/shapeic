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
    GmOverId,
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

pub(crate) fn standard_expression(
    model: &DeviceLut,
    expression: MosExpression,
) -> Result<Expr, LutError> {
    let parameter = |name| model.parameter_expression(name);

    match expression {
        MosExpression::Vsg => Ok(-Expr::axis(Axis::Vgs)),
        MosExpression::Vsb => Ok(-Expr::axis(Axis::Vbs)),
        MosExpression::Vsd => Ok(-Expr::axis(Axis::Vds)),
        MosExpression::GmOverId => Ok(parameter("gm")? / parameter("id")?),
        MosExpression::Vov => Ok(Expr::axis(Axis::Vgs) - parameter("vth")?),
        MosExpression::Vstar => Ok(2.0 * parameter("id")? / parameter("gm")?),
        MosExpression::CurrentDensity => Ok(parameter("id")? / width_expression(model)?),
        MosExpression::IntrinsicGain => Ok(parameter("gm")? / parameter("gds")?),
        MosExpression::TransitFrequency => {
            Ok(parameter("gm")? / (2.0 * std::f64::consts::PI * parameter("cgg")?))
        }
        MosExpression::EarlyVoltage => Ok(parameter("id")? / parameter("gds")?),
        MosExpression::InverseEarlyVoltage => Ok(parameter("gds")? / parameter("id")?),
        MosExpression::Rds => Ok(1.0 / parameter("gds")?),
        MosExpression::Vdsat => {
            for name in ["vdssat", "vdsat", "vsat"] {
                if model.parameters.contains_key(name) {
                    return parameter(name);
                }
            }
            Err(LutError::UnknownParameter {
                model: model.name().to_owned(),
                parameter: "vdssat/vdsat/vsat".to_owned(),
            })
        }
    }
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
