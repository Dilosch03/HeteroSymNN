import os
import sys
from unittest.mock import MagicMock
import sphinx_rtd_theme

# 1. Add the project root to the path
sys.path.insert(0, os.path.abspath('..'))

# 2. MOCK DEPENDENCIES
# This tells Sphinx: "If you can't find these libraries, just pretend they exist."
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Add every external library your project uses
MOCK_MODULES = [
    'numpy',
    'sympy', 
    'cupy', 
    'cupy.cuda', 
    'platformdirs',
    'scipy'
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
# -- Project information -----------------------------------------------------

project = 'HeteroSymNN'
copyright = '2025, Dilosch03'
author = 'Dilosch03'
release = '0.2.0'  # The full version, including alpha/beta/rc tags

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']