"""Build independent bilingual documentation and fragment-preserving legacy routes.

Only public documentation and examples are staged. StarWave is never imported.
"""
import argparse
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
SECTION = re.compile(r'<section id="([^"]+)">\s*(<h[1-6]\b.*?</h[1-6]>)', re.S)


def sections(text):
    return [(sid, html.unescape(re.sub('<[^>]+>', '', heading)).removesuffix('¶').strip())
            for sid, heading in SECTION.findall(text)]


def add_anchors(text, aliases=None):
    aliases = aliases or {}
    existing = set(re.findall(r' id="([^"]+)"', text))
    counter = iter(range(10000))
    def insert(match):
        number = next(counter)
        extra = ''.join(f'<span id="{html.escape(old)}" class="sw-anchor"></span>' for old, target in aliases.items() if target == f'sw-section-{number}' and old not in existing)
        return (f'<section id="{match[1]}" data-sw-section="sw-section-{number}">{extra}'
                f'<span id="sw-section-{number}" class="sw-anchor"></span>{match[2]}')
    return SECTION.sub(insert, text)


def redirect_page(relative, mapping):
    prefix = '../' * (len(Path(relative).parts) - 1)
    zh, en = prefix + 'zh/' + relative, prefix + 'en/' + relative
    # Location.replace keeps compatibility URLs out of browser history.
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StarWave · 中文 / English</title>
<link rel="canonical" href="{zh}">
<script>
const fragments = {json.dumps(mapping, ensure_ascii=True)};
let fragment = location.hash.slice(1);
try {{ fragment = decodeURIComponent(fragment); }} catch (_) {{}}
const target = new URL({json.dumps(zh)}, location.href);
target.search = location.search;
target.hash = fragments[fragment] || fragment;
location.replace(target.href);
</script></head><body>
<h1>StarWave 2.0.0</h1>
<p>文档已迁移至双语站点。Documentation is now available in two languages.</p>
<p><a href="{zh}" lang="zh-CN">继续阅读中文</a> · <a href="{en}" lang="en">Continue in English</a></p>
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
        source_files = {p.relative_to(source): p for p in source.rglob('*.md')
                        if 'en' not in p.relative_to(source).parts}
        if locale == 'en':
            # The English API body is a single source of truth in both locales.
            source_files.update({Path('api') / p.name: p for p in (PROJECT / 'docs' / 'api').glob('*.md')})
        source_paths[locale] = sorted(source_files)
        for relative, original in source_files.items():
            target = stage / locale / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = original.read_text(encoding='utf-8')
            if locale == 'zh' and relative == Path('index.md'):
                content = content.replace('\napi/index\n', '\nAPI 参考 <api/index>\n')
            target.write_text(content, encoding='utf-8')
        env = dict(os.environ, STARWAVE_DOCS_LANGUAGE=locale)
        subprocess.run([sys.executable, '-m', 'sphinx', '-n', '-W', '--keep-going', '-E',
                        '-b', 'html', '-c', str(PROJECT / 'docs'), '-d',
                        str(PROJECT / '_build' / 'doctrees' / locale),
                        str(stage / locale), str(destination / locale)], env=env, check=True)
    assert source_paths['zh'] == source_paths['en'], 'Translated page coverage differs'
    legacy = json.loads((PROJECT / 'tools' / 'legacy_anchors.json').read_text())
    api_aliases = json.loads((PROJECT / 'tools' / 'api_zh_anchors.json').read_text())
    for path in sorted((destination / 'zh').rglob('*.html')):
        relative = path.relative_to(destination / 'zh').as_posix()
        other = destination / 'en' / relative
        if not other.is_file():
            raise ValueError(f'Missing English counterpart: {relative}')
        texts = [path.read_text(encoding='utf-8'), other.read_text(encoding='utf-8')]
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
            mapping[entry['id']] = f'sw-section-{matches[0]}'
        texts[0] = texts[0].replace('aria-label="Main"', 'aria-label="主导航"')
        if relative.startswith('api/'):
            # English contents backlinks use numeric IDs that used to name Chinese
            # sections. Rename only those generated IDs in the Chinese context,
            # then restore historical IDs at their corresponding real sections.
            for old in api_aliases.get(relative, {}):
                if f' id="{old}"' in texts[0]:
                    texts[0] = texts[0].replace(f' id="{old}"', f' id="sw-toc-{old}"')
                    texts[0] = texts[0].replace(f'href="#{old}"', f'href="#sw-toc-{old}"')
            # Translate only Sphinx's generated field labels; shared prose is English.
            for old, new in [('参数', 'Parameters'), ('返回类型', 'Return type'), ('返回', 'Returns')]:
                texts[0] = re.sub(r'(<dt class="field-[^"]+">)' + old + r'(?=<span)', r'\g<1>' + new, texts[0])
            texts[0] = texts[0].replace('<div class="body" role="main">', '<div class="body" role="main" lang="en">')
        path.write_text(add_anchors(texts[0], api_aliases.get(relative)), encoding='utf-8')
        other.write_text(add_anchors(texts[1]), encoding='utf-8')
        old = destination / relative
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text(redirect_page(relative, mapping), encoding='utf-8')
    (destination / '.nojekyll').touch()
    print(f'Built {len(source_paths["zh"])} pages per language with independent search indexes and legacy routes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / '_build' / 'html')
    build(parser.parse_args().output)
