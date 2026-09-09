use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use super::Macro;

/// In-memory collection used to resolve macros by name.
#[derive(Clone, Debug, Default)]
pub struct MacroCatalog {
    macros: HashMap<String, Macro>,
}

impl MacroCatalog {
    /// Creates an empty macro catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a macro without silently replacing an existing definition.
    pub fn register(&mut self, macro_: Macro) -> Result<(), MacroCatalogError> {
        let name = macro_.name().to_owned();
        if self.macros.contains_key(&name) {
            return Err(MacroCatalogError::DuplicateMacro { name });
        }
        self.macros.insert(name, macro_);
        Ok(())
    }

    /// Builds a catalog from an iterator of macros.
    pub fn from_macros(macros: impl IntoIterator<Item = Macro>) -> Result<Self, MacroCatalogError> {
        let mut catalog = Self::new();
        for macro_ in macros {
            catalog.register(macro_)?;
        }
        Ok(catalog)
    }

    /// Returns a macro by name.
    pub fn get(&self, name: &str) -> Option<&Macro> {
        self.macros.get(name)
    }

    /// Returns whether a macro is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.macros.contains_key(name)
    }

    /// Returns all macros sorted by name.
    pub fn list(&self) -> Vec<&Macro> {
        let mut macros = self.macros.values().collect::<Vec<_>>();
        macros.sort_by(|left, right| left.name().cmp(right.name()));
        macros
    }
}

/// Errors produced while assembling a macro catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacroCatalogError {
    /// A macro name is already registered.
    DuplicateMacro { name: String },
}

impl fmt::Display for MacroCatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateMacro { name } => {
                write!(formatter, "macro '{name}' is already registered")
            }
        }
    }
}

impl Error for MacroCatalogError {}

#[cfg(test)]
mod tests {
    use crate::circuit::Circuit;

    use super::{Macro, MacroCatalog, MacroCatalogError};

    fn macro_(name: &str) -> Macro {
        Macro::new(
            name,
            Vec::new(),
            Circuit::builder().build(),
            Circuit::builder().build(),
        )
    }

    #[test]
    fn registers_resolves_and_lists_macros_by_name() {
        let catalog = MacroCatalog::from_macros([macro_("stage_b"), macro_("stage_a")]).unwrap();

        assert!(catalog.contains("stage_a"));
        assert_eq!(catalog.get("stage_b").map(Macro::name), Some("stage_b"));
        assert_eq!(
            catalog
                .list()
                .into_iter()
                .map(Macro::name)
                .collect::<Vec<_>>(),
            ["stage_a", "stage_b"]
        );
    }

    #[test]
    fn rejects_duplicate_macros_without_replacing_the_first() {
        let mut catalog = MacroCatalog::new();
        catalog.register(macro_("stage")).unwrap();

        assert_eq!(
            catalog.register(macro_("stage")),
            Err(MacroCatalogError::DuplicateMacro {
                name: "stage".to_owned(),
            })
        );
        assert_eq!(catalog.list().len(), 1);
    }
}
