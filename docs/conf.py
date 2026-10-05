"""Shared configuration for independent Chinese and English Sphinx builds."""
import os
from pathlib import Path

CONF_DIR = Path(__file__).resolve().parent
LOCALE = os.environ.get('STARWAVE_DOCS_LANGUAGE', 'zh')
if LOCALE not in {'zh', 'en'}:
    raise ValueError('STARWAVE_DOCS_LANGUAGE must be zh or en')
project = 'StarWave'
author = 'StarWave contributors'
copyright = '2026, StarWave contributors'
version = '2.0'
release = '2.0.0'
language = 'zh_CN' if LOCALE == 'zh' else 'en'
extensions = ['myst_parser', 'sphinx.ext.githubpages']
source_suffix = {'.md': 'markdown'}
root_doc = 'index'
exclude_patterns = ['en', 'requirements.txt', '_build', 'Thumbs.db', '.DS_Store']
myst_enable_extensions = ['colon_fence', 'fieldlist']
myst_heading_anchors = 3
nitpicky = True
# These annotations describe accepted public values, rather than local objects.
nitpick_ignore = [('py:class', x) for x in ('torch.Tensor', 'ScalarIllumination')]
html_theme = 'alabaster'
html_theme_options = {
    'description': 'CUDA 波传播 · PyTorch' if LOCALE == 'zh' else 'CUDA wave propagation · PyTorch',
    'fixed_sidebar': False, 'sidebar_collapse': False,
    'show_powered_by': False, 'show_related': False,
}
html_sidebars = {'**': ['about.html', 'navigation.html', 'searchbox.html']}
html_static_path = [str(CONF_DIR / '_static')]
templates_path = [str(CONF_DIR / '_templates')]
html_css_files = []  # Alabaster loads custom.css once.
html_js_files = ['language.js']
html_title = 'StarWave 2.0.0 使用手册' if LOCALE == 'zh' else 'StarWave 2.0.0 User Guide'
html_show_sourcelink = False
html_copy_source = False
html_show_sphinx = False
html_last_updated_fmt = None
html_search_language = language
html_baseurl = f'https://starrmoonn.github.io/StarWave-docs/{LOCALE}/'
html_context = {'sw_locale': LOCALE}


def page_context(app, pagename, templatename, context, doctree):
    prefix = '../' * (pagename.count('/') + 1)
    context['sw_zh_url'] = prefix + 'zh/' + pagename + '.html'
    context['sw_en_url'] = prefix + 'en/' + pagename + '.html'


def setup(app):
    app.connect('html-page-context', page_context)
