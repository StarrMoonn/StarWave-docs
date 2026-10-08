"""Build independent bilingual documentation and fragment-preserving legacy routes.

Only public documentation and examples are staged. StarWave is never imported.
"""
import argparse
import html
import json
import os
import posixpath
from pathlib import Path
import re
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
SECTION = re.compile(r'<section id="([^"]+)">\s*((?:<span id="[^"]+"></span>\s*)*)(<h[1-6]\b.*?</h[1-6]>)', re.S)


def sections(text):
    return [(sid, html.unescape(re.sub('<[^>]+>', '', re.sub(r'<a[^>]*class="headerlink"[^>]*>.*?</a>', '', heading))).strip())
            for sid, _, heading in SECTION.findall(text)]


def add_anchors(text, stable_anchors=None, section_numbers=None):
    counter = iter(range(10000))
    def insert(match):
        number = next(counter)
        section_number = section_numbers[number] if section_numbers else number
        stable = f' data-sw-anchor="{stable_anchors[number]}"' if stable_anchors else ''
        return (f'<section id="{match[1]}" data-sw-section="sw-section-{section_number}"{stable}>'
                f'<span id="sw-section-{section_number}" class="sw-anchor"></span>{match[2]}{match[3]}')
    return SECTION.sub(insert, text)



FWI_SECTIONS = ('fwi', 'fwi-run', 'fwi-configuration', 'fwi-loop', 'fwi-metrics', 'fwi-models', 'fwi-limits', 'fwi-wiring')
FWI_NUMBERS = (0, 4, 5, 2, 6, 7, 3, 1)

# Function subheadings merged into their propagator headings. Keep the original
# positional numbers and every published localized alias from before the merge.
USAGE_SECTIONS = ('usage', 'propagators', 'scalar', 'scalar-details',
                  'scalar-examples', 'scalar-time-sampling', 'scalar-memory',
                  'scalar-3d-example', 'scalar-notes', 'vrz', 'vrz-details',
                  'vrz-examples', 'vrz-notes', 'vti', 'vti-details',
                  'vti-examples', 'vti-notes', 'elastic', 'elastic-axes',
                  'elastic-returns', 'elastic-memory', 'elastic-examples',
                  'elastic-notes', 'elastic-conversions', 'native-runtime',
                  'native-status', 'prepare-native', 'native-notes',
                  'native-examples', 'prepare-elastic', 'other-exports')
# New content receives new numbers even when inserted before older sections.
USAGE_NUMBERS = (0, 1, 2, 4, 5, 31, 32, 33, 6, 7, 9, 10, 11, 12, 14, 15, 16,
                 23, 24, 25, 26, 27, 28, 29, 17, 18, 19, 20, 21, 30, 22)
MODELING_FRAGMENTS = {'modeling/vrz.md', 'modeling/vti.md'}
WAVE_SECTIONS = ('wave-propagation', 'wave-vrz', 'wave-vrz-details', 'wave-vrz-limits',
                 'wave-vti', 'wave-vti-details', 'wave-vti-limits')


RECONSTRUCTION_SECTIONS = ('reconstruction', 'reconstruction-scalar3d', 'reconstruction-state', 'reconstruction-tape',
                           'reconstruction-reverse', 'reconstruction-gradient',
                           'reconstruction-memory', 'reconstruction-scope',
                           'reconstruction-references')
RECONSTRUCTION_NUMBERS = (0, 8, 1, 2, 3, 4, 5, 6, 7)

def canonical_reconstruction_sections(text):
    """Keep all bilingual theory sections directly addressable."""
    assert len(sections(text)) == len(RECONSTRUCTION_SECTIONS), 'Reconstruction section structure changed'
    targets = iter(RECONSTRUCTION_SECTIONS)
    def replace(match):
        target = next(targets)
        aliases = dict.fromkeys([match[1], *re.findall(r'<span id="([^"]+)"', match[2])])
        spans = ''.join(f'<span id="{alias}"></span>' for alias in aliases if alias != target)
        heading = match[3].replace(f'href="#{match[1]}"', f'href="#{target}"')
        return f'<section id="{target}">{spans}{heading}'
    return SECTION.sub(replace, text)


def canonical_wave_sections(text):
    """Give the combined page stable, bilingual destinations for legacy routes."""
    assert len(sections(text)) == len(WAVE_SECTIONS), 'Wave propagation section structure changed'
    targets = iter(WAVE_SECTIONS)
    def replace(match):
        target = next(targets)
        aliases = dict.fromkeys([match[1], *re.findall(r'<span id="([^"]+)"', match[2])])
        spans = ''.join(f'<span id="{alias}"></span>' for alias in aliases if alias != target)
        heading = match[3].replace(f'href="#{match[1]}"', f'href="#{target}"')
        return f'<section id="{target}">{spans}{heading}'
    return SECTION.sub(replace, text)


def canonical_usage_sections(text, locale):
    aliases = json.loads((PROJECT / 'tools' / 'usage_anchors.json').read_text())[locale]
    assert len(sections(text)) == len(USAGE_SECTIONS), 'Usage section structure changed'
    targets = iter(zip(USAGE_SECTIONS, USAGE_NUMBERS))
    def replace(match):
        target, number = next(targets)
        # Discard regenerated idN aliases, whose meaning can shift when headings
        # disappear. Recreate only the published aliases with stable meanings.
        spans = ''.join(f'<span id="{alias}"></span>'
                        for alias, destination in aliases.items()
                        if destination == target and alias not in {target, f'sw-section-{number}'})
        heading = match[3].replace(f'href="#{match[1]}"', f'href="#{target}"')
        return f'<section id="{target}">{spans}{heading}'
    return SECTION.sub(replace, text)


def canonical_fwi_sections(text, locale):
    """Preserve original localized and positional FWI bookmarks after expansion."""
    aliases = {'fwi-wiring': 'id1', 'fwi-loop': 'id2', 'fwi-limits': 'id3'} if locale == 'zh' else {
        'fwi': 'introduction-to-fwi', 'fwi-wiring': 'run-one-update',
        'fwi-loop': 'key-points-for-the-loop', 'fwi-limits': 'validation-limits'}
    counter = iter(FWI_SECTIONS)
    def replace(match):
        target = next(counter)
        alias = f'<span id="{aliases[target]}"></span>' if target in aliases else ''
        heading = match[3].replace(f'href="#{match[1]}"', f'href="#{target}"')
        return f'<section id="{target}">{alias}{heading}'
    return SECTION.sub(replace, text)


def add_page_toc(text, locale):
    """Keep headings in the article rather than expandable sidebar branches."""
    headings = sections(text)
    if len(headings) < 2:
        return text
    label = '本页内容' if locale == 'zh' else 'On this page'
    links = ''.join(f'<li><a href="#{html.escape(sid, quote=True)}">{html.escape(title)}</a></li>'
                    for sid, title in headings[1:])
    toc = f'<nav class="sw-page-toc" aria-label="{label}"><p><strong>{label}</strong></p><ul>{links}</ul></nav>'
    return SECTION.sub(lambda match: match[0] + toc, text, count=1)


def redirect_page(relative, mapping, *, target_relative=None, locale='zh', default_fragment=''):
    target_relative = target_relative or relative
    parent = posixpath.dirname(relative)
    targets = {lang: posixpath.relpath(lang + '/' + target_relative, parent or '.')
               for lang in ('zh', 'en')}
    target = targets[locale]
    zh = targets['zh'] + ('#' + default_fragment if default_fragment else '')
    en = targets['en'] + ('#' + default_fragment if default_fragment else '')
    title = '文档已迁移' if locale == 'zh' else 'Documentation moved'
    language = 'zh-CN' if locale == 'zh' else 'en'
    message = '请使用下方链接继续阅读。' if locale == 'zh' else 'Use the links below to continue reading.'
    # Location.replace keeps compatibility URLs out of browser history.
    return f'''<!doctype html>
<html lang="{language}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, follow">
<title>{title} · StarWave</title>
<link rel="canonical" href="{target}">
<script>
const fragments = {json.dumps(mapping, ensure_ascii=True)};
let fragment = location.hash.slice(1);
try {{ fragment = decodeURIComponent(fragment); }} catch (_) {{}}
const target = new URL({json.dumps(target)}, location.href);
target.search = location.search;
target.hash = fragments[fragment] || fragment || {json.dumps(default_fragment)};
location.replace(target.href);
</script></head><body>
<h1>{title}</h1>
<p>{message}</p>
<p><a href="{zh}" lang="zh-CN" data-sw-language="zh">中文</a> · <a href="{en}" lang="en" data-sw-language="en">Continue in English</a></p>
</body></html>'''


def build(destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stage = PROJECT / '_build' / 'source'
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copytree(PROJECT / 'examples', stage / 'examples')
    source_paths = {}
    for locale, source in [('zh', PROJECT / 'docs'), ('en', PROJECT / 'docs' / 'en')]:
        source_paths[locale] = sorted(p.relative_to(source) for p in source.rglob('*.md')
                                      if 'en' not in p.relative_to(source).parts)
        for relative in source_paths[locale]:
            target = stage / locale / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        # Keep these files available to MyST include, but publish their content
        # only once, in Wave propagation. Old URLs are emitted as redirects.
        source_paths[locale] = [p for p in source_paths[locale] if p.as_posix() not in MODELING_FRAGMENTS]
        shutil.copytree(PROJECT / 'docs' / '_static', stage / locale / '_static')
        env = dict(os.environ, STARWAVE_DOCS_LANGUAGE=locale)
        subprocess.run([sys.executable, '-m', 'sphinx', '-n', '-W', '--keep-going', '-E',
                        '-b', 'html', '-c', str(PROJECT / 'docs'), '-d',
                        str(PROJECT / '_build' / 'doctrees' / locale),
                        str(stage / locale), str(destination / locale)], env=env, check=True)
    assert source_paths['zh'] == source_paths['en'], 'Translated page coverage differs'
    legacy = json.loads((PROJECT / 'tools' / 'legacy_anchors.json').read_text())
    pages = {path.with_suffix('.html').as_posix() for path in source_paths['zh']} | {'genindex.html', 'search.html'}
    for relative in sorted(pages):
        path = destination / 'zh' / relative
        other = destination / 'en' / relative
        if not other.is_file():
            raise ValueError(f'Missing English counterpart: {relative}')
        texts = [path.read_text(encoding='utf-8'), other.read_text(encoding='utf-8')]
        if relative == 'inversion/fwi.html':
            texts = [canonical_fwi_sections(text, locale) for text, locale in zip(texts, ('zh', 'en'))]
        if relative == 'usage.html':
            texts = [canonical_usage_sections(text, locale) for text, locale in zip(texts, ('zh', 'en'))]
        if relative == 'modeling/wave-propagation.html':
            texts = [canonical_wave_sections(text) for text in texts]
        if relative == 'modeling/reconstruction.html':
            texts = [canonical_reconstruction_sections(text) for text in texts]
        pairs = [sections(text) for text in texts]
        assert len(pairs[0]) == len(pairs[1]), f'Heading parity differs: {relative}'
        mapping = {}
        for entry in legacy.get(relative, []):
            if 'target' in entry:
                assert int(entry['target'].rsplit('-', 1)[1]) < len(pairs[0])
                mapping[entry['id']] = entry['target']
                continue
            matches = [i for i, (_, title) in enumerate(pairs[0]) if title == entry['title']]
            if len(matches) != 1:
                raise ValueError(f"Legacy heading missing/ambiguous: {relative} {entry['title']}")
            number = FWI_NUMBERS[matches[0]] if relative == 'inversion/fwi.html' else matches[0]
            mapping[entry['id']] = f'sw-section-{number}'
        texts[0] = texts[0].replace('aria-label="Main"', 'aria-label="主导航"')
        if relative not in {'usage.html', 'index.html'}:
            texts = [add_page_toc(text, locale) for text, locale in zip(texts, ('zh', 'en'))]
        stable = None
        if relative == 'usage.html':
            stable = USAGE_SECTIONS
            for source in (PROJECT / 'docs', PROJECT / 'docs' / 'en'):
                source_sections = ['usage'] + re.findall(r'^\(([^)]+)\)=\n#+ ', (source / 'usage.md').read_text(), re.M)
                assert source_sections == list(stable), f'Usage source order differs from stable section map: {source.name}'
            assert len(stable) == len(pairs[0]), 'Usage anchors must cover every section'
        numbers = USAGE_NUMBERS if relative == 'usage.html' else None
        if relative == 'inversion/fwi.html':
            numbers = FWI_NUMBERS
            stable = FWI_SECTIONS
        if relative == 'modeling/wave-propagation.html':
            stable = WAVE_SECTIONS
        if relative == 'modeling/reconstruction.html':
            stable = RECONSTRUCTION_SECTIONS
            numbers = RECONSTRUCTION_NUMBERS
        rendered = [add_anchors(text, stable, numbers) for text in texts]
        if relative.startswith('examples/'):
            rendered = [text.replace('<section ', '<section class="sw-example-page" ', 1) for text in rendered]
        path.write_text(rendered[0], encoding='utf-8')
        other.write_text(rendered[1], encoding='utf-8')
        old = destination / relative
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text(redirect_page(relative, mapping), encoding='utf-8')
    api_redirects = json.loads((PROJECT / 'tools' / 'api_redirects.json').read_text())
    for source, routes in api_redirects.items():
        for relative, route in routes.items():
            relative = relative if source == 'root' else source + '/' + relative
            old = destination / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_text(redirect_page(relative, route['fragments'], target_relative='usage.html',
                                         locale='zh' if source == 'root' else source,
                                         default_fragment=route['default']), encoding='utf-8')
    modeling_redirects = json.loads((PROJECT / 'tools' / 'modeling_redirects.json').read_text())
    for source, routes in modeling_redirects.items():
        for relative, route in routes.items():
            relative = relative if source == 'root' else source + '/' + relative
            old = destination / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_text(redirect_page(relative, route['fragments'],
                                         target_relative='modeling/wave-propagation.html',
                                         locale='zh' if source == 'root' else source,
                                         default_fragment=route['default']), encoding='utf-8')
    (destination / '.nojekyll').touch()
    print(f'Built {len(source_paths["zh"])} pages per language with independent search indexes and legacy routes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / '_build' / 'html')
    build(parser.parse_args().output)
