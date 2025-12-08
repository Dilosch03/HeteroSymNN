import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'HeteroSymNN'
copyright = '2024, Dilosch03'
author = 'Dilosch03'
release = '0.2.0'

extensions = [
    'sphinx.ext.autodoc',     
    'sphinx.ext.autosummary',  
    'sphinx.ext.napoleon',     
    'sphinx_rtd_theme',        
]

# Tell Sphinx to mock modules that are not available on Read the Docs
autodoc_mock_imports = ["cupy", "matplotlib"]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']