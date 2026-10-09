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
version = '6.0'
release = '6.0.0'
language = 'zh_CN' if LOCALE == 'zh' else 'en'
extensions = ['myst_parser', 'sphinx.ext.githubpages', 'sphinx.ext.mathjax']
source_suffix = {'.md': 'markdown'}
root_doc = 'index'
exclude_patterns = ['en', 'requirements.txt', '_build', 'Thumbs.db', '.DS_Store',
                    'modeling/vrz.md', 'modeling/vti.md']
myst_enable_extensions = ['colon_fence', 'fieldlist']
myst_heading_anchors = 3
nitpicky = True
# These annotations describe accepted public values, rather than local objects.
nitpick_ignore = [('py:class', x) for x in ('torch.Tensor', 'ScalarIllumination', 'Optional', 'Union', 'Sequence', 'Literal', 'Tuple', 'common.Callback')]
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 1,
    'titles_only': True,
    'style_nav_header_background': '#487eae',
}
html_static_path = [str(CONF_DIR / '_static')]
templates_path = [str(CONF_DIR / '_templates')]
html_css_files = ['custom.css']
html_favicon = str(CONF_DIR / '_static' / 'brand' / 'starwave-favicon.png')
html_js_files = ['language.js']
html_title = 'StarWave 6.0.0 使用手册' if LOCALE == 'zh' else 'StarWave 6.0.0 User Guide'
html_show_sourcelink = False
html_copy_source = False
html_show_sphinx = True
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
