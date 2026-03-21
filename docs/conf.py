import sys, os
sys.path.insert(0, os.path.abspath(".."))

project = "MTF"
author = "MTF Contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

myst_enable_extensions = ["colon_fence", "deflist"]

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/FaroutYLq/mtf",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
}
html_title = "MTF — My Theorist Friend"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

exclude_patterns = ["_build"]
