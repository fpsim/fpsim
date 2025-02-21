# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import sciris as sc
import fpsim as fp

# Set environment
os.environ['SPHINX_BUILD'] = 'True' # This is used so fp.options.set('jupyter') doesn't reset the Matplotlib renderer
os.environ['FPSIM_WARNINGS'] = 'error' # Don't let warnings pass in the tutorials
on_rtd = os.environ.get('READTHEDOCS') == 'True'


# -- Project information -----------------------------------------------------


# The full version, including alpha/beta/rc tags
release = fp.__version__


# -- General configuration ---------------------------------------------------


napoleon_google_docstring = True

# Configure autosummary
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_ignore_module_all = False # Respect __all__
autodoc_member_order = 'bysource' # Keep original ordering
add_module_names = False  # NB, does not work
autodoc_inherit_docstrings = False # Stops sublcasses from including docs from parent classes

# Add any paths that contain templates here, relative to this directory.

html_favicon = "images/favicon.ico"
html_static_path = ['_static']
html_baseurl = "https://docs.idmod.org/projects/fpsim/en/latest"
html_context = {
    'rtd_url': 'https://docs.idmod.org/projects/fpsim/en/latest',
    "versions_dropdown": {
        "latest": "devel (latest)",
        "stable": "current (stable)",
    },
    "default_mode": "light",
}
# Add any extra paths that contain custom files
if not on_rtd:
    html_extra_path = ['robots.txt']


# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_last_updated_fmt = '%Y-%b-%d'
html_show_sourcelink = True
html_show_sphinx = False
html_copy_source = False
htmlhelp_basename = 'FPsim'

# Add customizations
def setup(app):
    app.add_css_file("theme_overrides.css")


# Modify this to not rerun the Jupyter notebook cells -- usually set by build_docs
nb_ex_default = ['auto', 'never'][0]
nb_ex = os.getenv('NBSPHINX_EXECUTE')
if not nb_ex: nb_ex = nb_ex_default
print(f'\n\nBuilding Jupyter notebooks with build option: {nb_ex}\n\n')
nbsphinx_execute = nb_ex

# OpenSearch options
html_use_opensearch = 'docs.idmod.org/projects/fpsim/en/latest'

# -- RTD Sphinx search for searching across the entire domain, default child -------------
if os.environ.get('READTHEDOCS') == 'True':

    search_project_parent = "institute-for-disease-modeling-idm"
    search_project = os.environ["READTHEDOCS_PROJECT"]
    search_version = os.environ["READTHEDOCS_VERSION"]

    rtd_sphinx_search_default_filter = f"subprojects:{search_project}/{search_version}"

    rtd_sphinx_search_filters = {
        "Search this project": f"project:{search_project}/{search_version}",
        "Search all IDM docs": f"subprojects:{search_project_parent}/{search_version}",
    }
