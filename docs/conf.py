import os
import sys
from unittest.mock import MagicMock

# Signal to the code that we are building documentation
os.environ["SPHINX_BUILD"] = "True"

# 1. Add the project root to the path
sys.path.insert(0, os.path.abspath('..'))

# Mock platformdirs manually because pathlib.Path fails with standard MagicMock
mock_platformdirs = MagicMock()
mock_platformdirs.user_cache_dir.return_value = "/tmp/mock_cache"
sys.modules['platformdirs'] = mock_platformdirs

# 2. MOCK DEPENDENCIES
# Use autodoc_mock_imports to let Sphinx handle mocking gracefully
autodoc_mock_imports = [
    'numpy',
    'sympy', 
    'cupy', 
    'scipy'
]
# -- Project information -----------------------------------------------------

project = 'HeteroSymNN'
copyright = '2025, Dilosch03'
author = 'Dilosch03'
release = '0.3.0rc4'  # The full version, including alpha/beta/rc tags

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

# Prevent Sphinx from expanding type aliases and defaults
autodoc_preserve_defaults = True
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# -- Custom Roles ------------------------------------------------------------
from sphinx.roles import XRefRole
from docutils import nodes

def resolve_customref(app, env, node, contnode):
    if node.get('reftype') == 'customref':
        std = env.get_domain('std')
        node['reftype'] = 'ref'
        res = std.resolve_xref(env, node.get('refdoc', ''), app.builder, 'ref', node['reftarget'], node, contnode)
        node['reftype'] = 'customref'
        
        if res is not None:
            if len(res) > 0 and isinstance(res[0], nodes.inline):
                literal = nodes.literal('', res[0].astext(), classes=['xref', 'py', 'py-obj'])
                res.replace(res[0], literal)
            return res
    return None

def setup(app):
    """
    Registers custom roles for Sphinx.
    """
    app.add_role('customref', XRefRole(lowercase=True, warn_dangling=True))
    app.connect('missing-reference', resolve_customref)