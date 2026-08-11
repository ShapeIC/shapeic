# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "Shape-IC User Guide"
author = "Daniel Arevalos"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]


# -- Options for HTML output

html_theme = "furo"

# html_extra_path = ['../TOP.html']

# -- Options for EPUB output
epub_show_urls = "footnote"
