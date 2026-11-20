# Configuration file for the Sphinx documentation builder.
#
# For a full list of options see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup ---------------------------------------------------------
#
# CompStats is installed (editable) by the builds that render these docs
# (.github/workflows/test.yml and website.yml), so conf.py only imports the
# installed package to read its __version__.

import CompStats

# -- Project information -------------------------------------------------

project = 'CompStats'
copyright = '2024, Sergio Nava-Muñoz, Mario Graff, Hugo Jair Escalante'
author = 'Sergio Nava-Muñoz, Mario Graff, Hugo Jair Escalante'
version = CompStats.__version__
release = version

rst_epilog = f"""
.. |compstats_version| replace:: {CompStats.__version__}
"""

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

source_suffix = '.rst'
master_doc = 'index'
language = 'en'
exclude_patterns = []

pygments_style = 'sphinx'
highlight_language = 'python'
autodoc_member_order = 'bysource'
autodoc_class_signature = 'separated'
add_function_parentheses = False
add_module_names = False

# -- Options for HTML output ----------------------------------------------

html_theme = 'furo'
html_logo = 'ingeotec.png'
htmlhelp_basename = 'CompStatsdoc'
