# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess

# Doxygen
doxygen_cmd = "doxygen Doxyfile.in"  # or Doxyfile.in if that's the correct filename

# Create the doxygen subdirectory if it doesn't exist
if os.path.exists("_build/doxygen") is False:
    os.makedirs("_build/doxygen")

if os.environ.get("NODOXYGEN") is not None:
    print("Skipping Doxygen build as NODOXYGEN environment variable is set")
    result = 0
else:
    result = subprocess.call(doxygen_cmd, shell=True)

if result != 0:
    print(f"Error: Doxygen command '{doxygen_cmd}' failed with code {result}")
    sys.exit(1)

# Check if output directory exists
if not os.path.exists("_build/doxygen/xml"):
    print("Error: Doxygen XML output directory '_build/doxygen/xml' not found")
    print("Please check Doxygen configuration and make sure it runs correctly")
    sys.exit(1)

# -- Project information -----------------------------------------------------

project = "SPECFEM++"
copyright = "2023, Rohit Kakodkar"
author = "Rohit Kakodkar"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_sitemap",
    "sphinx.ext.inheritance_diagram",
    "sphinx_design",
    "breathe",
    "sphinx_copybutton",
]

# Adding this to avoid the WARNING: duplicate label warning
# autosectionlabel_prefix_document = True
supress_warnings = ["*duplicate*"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

highlight_language = "c++"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
# html_theme_options = {
#     "canonical_url": "",
#     "analytics_id": "",  #  Provided by Google in your dashboard
#     "display_version": True,
#     "prev_next_buttons_location": "bottom",
#     "style_external_links": False,
#     "logo_only": False,
#     # Toc options
#     "collapse_navigation": True,
#     "sticky_navigation": True,
#     "navigation_depth": 4,
#     "includehidden": True,
#     "titles_only": False,
# }
# html_logo = ''
github_url = "https://github.com/PrincetonUniversity/SPECFEMPP"
html_baseurl = "https://specfem2d-kokkos.readthedocs.io/"

# External links configuration
extlinks = {
    "issue": (f"{github_url}/issues/%s", "Issue #%s"),
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/scrollable_code_blocks.css",
    "css/center_align_table.css",
]


# -- Breathe configuration -------------------------------------------------

breathe_projects = {"specfem++": "_build/doxygen/xml"}
breathe_default_project = "specfem++"
breathe_default_members = ()
breathe_doxygen_config_options = {"PREDEFINED": "KOKKOS_INLINE_FUNCTION="}
breathe_show_define_initializer = True
breathe_show_include = True
breathe_template_parameters = True
breathe_separate_member_pages = True
