# docs/conf.py
import os
import sys
from datetime import datetime

# assume that quake has been installed with pip and the bindings are located in the pip site-packages
import quake

quake_path = os.path.dirname(os.path.dirname(quake.__file__))
print(f"Adding quake path: {quake_path}")

# Add the path to your bindings (adjust as needed)
sys.path.insert(0, quake_path)
# -- Project information -----------------------------------------------------
project = "Quake"
author = "Jason Mohoney"
copyright = f"{datetime.now().year}, {author}"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Automatically extract docstrings from Python
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinxcontrib.mermaid",  # Add support for Mermaid diagrams
    "sphinx.ext.graphviz",  # Add support for Graphviz diagrams
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def setup(app):
    app.add_css_file("css/modify.css")
