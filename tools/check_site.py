"""Check a built documentation site and the deliberately small source package."""
import ast
import hashlib
import struct
import json
import math
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlsplit
from build_docs import USAGE_SECTIONS, USAGE_NUMBERS
from check_presentation_asset import ASSET_PATH, check as check_presentation
from check_scalar3d_examples import approved_assets, check as check_scalar3d
from check_sls_examples import check as check_sls


RECONSTRUCTION_SECTIONS = ('reconstruction', 'reconstruction-scalar3d', 'reconstruction-state',
                           'reconstruction-tape', 'reconstruction-reverse',
                           'reconstruction-gradient', 'reconstruction-memory',
                           'reconstruction-scope', 'reconstruction-references')
RECONSTRUCTION_FIGURES = ('flow', 'domain', 'timeline', 'memory')
RECONSTRUCTION_ASSETS = {f'{name}-{locale}.svg'
                         for name in RECONSTRUCTION_FIGURES for locale in ('zh', 'en')}


class HTMLTree(HTMLParser):
    """Small, dependency-free tree for semantic checks of generated HTML."""
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
            'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.root = ET.Element('document')
        self.stack = [self.root]
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        node = ET.SubElement(self.stack[-1], tag, dict(attrs))
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        ET.SubElement(self.stack[-1], tag, dict(attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        parent = self.stack[-1]
        if len(parent):
            parent[-1].tail = (parent[-1].tail or '') + data
        else:
            parent.text = (parent.text or '') + data


def element_text(element):
    return ' '.join(''.join(element.itertext()).split()) if element is not None else ''


def reconstruction_svg_errors(data, name, locale):
    """Validate original vector diagrams without executing or loading resources."""
    errors = []
    try:
        text = data.decode('utf-8')
        if re.search(r'<!DOCTYPE|<!ENTITY|<\?(?!xml\s)', text, re.I):
            errors.append(f'Active or external SVG declaration: {name}')
        svg = ET.fromstring(data)
    except (ET.ParseError, UnicodeDecodeError):
        return [f'Invalid reconstruction SVG: {name}']
    if svg.tag != '{http://www.w3.org/2000/svg}svg':
        errors.append(f'Invalid reconstruction SVG root: {name}')
    try:
        box = [float(n) for n in re.split(r'[\s,]+', svg.get('viewBox', '').strip())]
        if len(box) != 4 or not all(math.isfinite(n) for n in box) or min(box[2:]) <= 0:
            raise ValueError
    except ValueError:
        errors.append(f'Invalid reconstruction SVG viewBox: {name}')
    ids = [element.get('id') for element in svg.iter() if element.get('id')]
    if len(ids) != len(set(ids)):
        errors.append(f'Duplicate reconstruction SVG IDs: {name}')
    labelled = svg.get('aria-labelledby', '').split()
    if svg.get('role') != 'img' or not labelled or any(target not in ids for target in labelled):
        errors.append(f'Missing accessible reconstruction SVG name: {name}')
    for tag in ('title', 'desc'):
        nodes = svg.findall('{http://www.w3.org/2000/svg}' + tag)
        value = element_text(nodes[0]) if len(nodes) == 1 else ''
        if not value or not nodes[0].get('id') or nodes[0].get('id') not in labelled:
            errors.append(f'Missing accessible reconstruction SVG {tag}: {name}')
        if value and bool(re.search(r'[\u4e00-\u9fff]', value)) != (locale == 'zh'):
            errors.append(f'Unlocalized reconstruction SVG {tag}: {name}')
    # A strict vector-only allowlist excludes scripts, images, embedded HTML,
    # animation and external resources. Local marker/clip/glyph reuse is safe.
    allowed = {'svg', 'g', 'defs', 'title', 'desc', 'path', 'rect', 'circle',
               'ellipse', 'line', 'polyline', 'polygon', 'text', 'tspan', 'marker',
               'pattern', 'clipPath', 'mask', 'linearGradient', 'radialGradient',
               'stop', 'style', 'use'}
    labels = []
    for element in svg.iter():
        tag = element.tag.rsplit('}', 1)[-1]
        if not element.tag.startswith('{http://www.w3.org/2000/svg}') or tag not in allowed:
            errors.append(f'Unsafe reconstruction SVG element: {name}/{tag}')
        if tag == 'style':
            style = element_text(element)
            if '\\' in style or re.search(r'@import|expression\s*\(', style, re.I):
                errors.append(f'Active reconstruction SVG style: {name}')
            for target in re.findall(r'url\s*\((.*?)\)', style, re.I):
                if not re.fullmatch(r'#[A-Za-z_][A-Za-z0-9_.:-]*', target) or target[1:] not in ids:
                    errors.append(f'External reconstruction SVG style reference: {name}')
        for key, value in element.attrib.items():
            local_key = key.rsplit('}', 1)[-1].lower()
            if local_key.startswith('on') or local_key in {'src', 'base'}:
                errors.append(f'Active reconstruction SVG attribute: {name}/{key}')
            if local_key == 'href' and (not value.startswith('#') or value[1:] not in ids):
                errors.append(f'External or missing reconstruction SVG reference: {name}/{key}')
            if (local_key == 'style' and '\\' in value) or re.search(r'expression\s*\(|@import', value, re.I):
                errors.append(f'Active reconstruction SVG style: {name}/{key}')
            for target in re.findall(r'url\s*\((.*?)\)', value, re.I):
                if not re.fullmatch(r'#[A-Za-z_][A-Za-z0-9_.:-]*', target) or target[1:] not in ids:
                    errors.append(f'External reconstruction SVG reference: {name}/{key}')
        if tag == 'text' and element_text(element):
            labels.append(element_text(element))
        if tag == 'g' and element.get('aria-label') and element.get('data-font-size'):
            labels.append(element.get('aria-label'))
            try:
                size = float(element.get('data-font-size'))
                if not math.isfinite(size) or size <= 0 or not any(
                        node.tag.rsplit('}', 1)[-1] == 'path' and node.get('d')
                        for node in element.iter()):
                    raise ValueError
            except ValueError:
                errors.append(f'Invalid outlined reconstruction label: {name}')
    if len(labels) < 4:
        errors.append(f'Missing meaningful reconstruction SVG labels: {name}')
    if locale == 'en' and any(re.search(r'[\u4e00-\u9fff]', value) for value in labels):
        errors.append(f'Chinese labels in English reconstruction SVG: {name}')
    if locale == 'zh' and not any(re.search(r'[\u4e00-\u9fff]', value) for value in labels):
        errors.append(f'Missing Chinese reconstruction SVG labels: {name}')
    return errors


def check_reconstruction(project, root):
    """Guard the bilingual chapter, original figures, and API round-trip links."""
    errors = []
    assets = project / 'docs' / '_static' / 'reconstruction'
    if not assets.is_dir() or {path.name for path in assets.iterdir()} != RECONSTRUCTION_ASSETS:
        errors.append('Unexpected or incomplete reconstruction asset set')
    seen_assets = {}
    for name in sorted(RECONSTRUCTION_ASSETS):
        path = assets / name
        if not path.is_file():
            errors.append(f'Missing reconstruction SVG: {name}')
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen_assets:
            errors.append(f'Repeated reconstruction SVG content: {name}/{seen_assets[digest]}')
        seen_assets[digest] = name
        errors.extend(reconstruction_svg_errors(data, name, path.stem.rsplit('-', 1)[-1]))
        for locale in ('zh', 'en'):
            published = root / locale / '_static' / 'reconstruction' / name
            if not published.is_file() or published.read_bytes() != data:
                errors.append(f'Published reconstruction SVG differs: {locale}/{name}')
    for locale in ('zh', 'en'):
        chapter = root / locale / 'modeling' / 'reconstruction.html'
        if not chapter.is_file():
            errors.append(f'Missing reconstruction chapter: {locale}')
            continue
        tree = HTMLTree(chapter.read_text()).root
        main = tree.find('.//*[@role="main"]')
        if main is None:
            errors.append(f'Missing reconstruction article: {locale}')
            continue
        title = '波场反传重建' if locale == 'zh' else 'Wavefield Reconstruction'
        headings = main.findall('.//h1')
        if len(headings) != 1 or (headings[0].text or '').strip() != title:
            errors.append(f'Wrong reconstruction title: {locale}')
        sections = main.findall('.//section')
        if tuple(section.get('data-sw-anchor') for section in sections) != RECONSTRUCTION_SECTIONS:
            errors.append(f'Missing stable reconstruction sections: {locale}')
        for section, target in zip(sections, RECONSTRUCTION_SECTIONS):
            if section.get('id') != target:
                errors.append(f'Reconstruction ID/anchor mismatch: {locale}/{target}')
            heading = next((e for e in section if re.fullmatch('h[1-6]', e.tag)), None)
            if heading is None or not any(a.get('href') == '#' + target for a in heading.iter('a')):
                errors.append(f'Missing reconstruction heading permalink: {locale}/{target}')
            body = ' '.join(element_text(e) for e in section if e.tag not in {'section', 'nav', 'span', 'h1', 'h2', 'h3'})
            if len(body) < (60 if locale == 'zh' else 140):
                errors.append(f'Incomplete reconstruction explanation: {locale}/{target}')
        if len(element_text(main)) < (1400 if locale == 'zh' else 3500):
            errors.append(f'Reconstruction chapter lacks substantive text: {locale}')
        tocs = [nav for nav in main.iter('nav') if 'sw-page-toc' in nav.get('class', '').split()]
        if len(tocs) != 1 or [a.get('href') for a in tocs[0].iter('a')] != ['#' + target for target in RECONSTRUCTION_SECTIONS[1:]]:
            errors.append(f'Invalid reconstruction local TOC: {locale}')
        figures = [figure for figure in main.iter('figure') if 'sw-reconstruction-figure' in figure.get('class', '').split()]
        if len(figures) != 4:
            errors.append(f'Incomplete reconstruction figures: {locale}')
        for figure, name in zip(figures, RECONSTRUCTION_FIGURES):
            images = figure.findall('.//img')
            if len(images) != 1:
                errors.append(f'Invalid reconstruction figure image: {locale}/{name}')
                continue
            image = images[0]
            source = urlsplit(image.get('src', ''))
            expected = root / locale / '_static' / 'reconstruction' / f'{name}-{locale}.svg'
            if source.scheme or source.netloc or (chapter.parent / unquote(source.path)).resolve() != expected:
                errors.append(f'Nonlocal or wrong reconstruction figure: {locale}/{name}')
            alt = image.get('alt', '').strip()
            caption = element_text(figure.find('figcaption'))
            for field, value, minimum in (('alt', alt, 20 if locale == 'zh' else 40),
                                           ('caption', caption, 40 if locale == 'zh' else 100)):
                if len(value) < minimum or bool(re.search(r'[\u4e00-\u9fff]', value)) != (locale == 'zh'):
                    errors.append(f'Missing detailed localized reconstruction {field}: {locale}/{name}')
        usage = root / locale / 'usage.html'
        if usage.is_file():
            usage_tree = HTMLTree(usage.read_text()).root
            memory = usage_tree.find('.//section[@id="elastic-memory"]')
            if memory is None or not any(
                    urlsplit(a.get('href', '')).path == 'modeling/reconstruction.html'
                    and urlsplit(a.get('href', '')).fragment in ('', *RECONSTRUCTION_SECTIONS)
                    for a in memory.iter('a')):
                errors.append(f'Missing Usage elastic reconstruction backlink: {locale}')
        if not any(urlsplit(a.get('href', '')).path == '../usage.html'
                   and urlsplit(a.get('href', '')).fragment in {'elastic', 'elastic-memory', 'starwave.elastic'}
                   for a in main.iter('a')):
            errors.append(f'Missing reconstruction elastic Usage link: {locale}')
        expected_label = '波场反传重建' if locale == 'zh' else 'Reconstruction'
        sidebar = next((e for e in tree.iter('div') if 'wy-menu-vertical' in e.get('class', '').split()), None)
        sidebar_links = [] if sidebar is None else [a for a in sidebar.iter('a')
            if not urlsplit(a.get('href', '')).scheme and not urlsplit(a.get('href', '')).netloc
            and ((chapter.parent / unquote(urlsplit(a.get('href', '')).path)).resolve()
                 if urlsplit(a.get('href', '')).path else chapter) == chapter]
        if len(sidebar_links) != 1 or element_text(sidebar_links[0]) != expected_label:
            errors.append(f'Missing or mislabeled reconstruction sidebar chapter: {locale}')
        if sidebar is not None:
            group = None
            found = False
            for child in sidebar:
                if child.tag == 'p' and 'caption' in child.get('class', '').split():
                    group = element_text(child)
                if child.tag == 'ul' and any(a in sidebar_links for a in child.iter('a')):
                    found = group == ('正演模拟' if locale == 'zh' else 'Forward Modeling')
            if not found:
                errors.append(f'Reconstruction is not under Forward Modeling: {locale}')
        references = main.find('.//section[@id="reconstruction-references"]')
        if references is None or not any(urlsplit(a.get('href', '')).scheme == 'https' for a in references.iter('a')):
            errors.append(f'Missing reconstruction reference links: {locale}')
    return errors


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.duplicate_ids = set()
        self.id_sections = {}
        self.sections = []
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'section':
            self.sections.append(attrs.get('data-sw-section'))
        if "id" in attrs:
            if attrs['id'] in self.ids:
                self.duplicate_ids.add(attrs['id'])
            self.ids.add(attrs["id"])
            self.id_sections[attrs['id']] = self.sections[-1] if self.sections else None
        if tag == "a" and "name" in attrs:
            self.ids.add(attrs["name"])
        for key in ("href", "src") + (("data",) if tag == "object" else ()):
            if key in attrs:
                self.links.append(attrs[key])

    def handle_endtag(self, tag):
        if tag == 'section' and self.sections:
            self.sections.pop()


def check(root):
    project = Path(__file__).resolve().parents[1]
    root = root.resolve()
    errors = []
    errors.extend(check_presentation(root))
    errors.extend(check_reconstruction(project, root))
    errors.extend(check_scalar3d(root))
    errors.extend(check_sls(root))
    pages = {p: Page(p.read_text(encoding="utf-8")) for p in root.rglob("*.html")}
    if not pages or not (root / "index.html").is_file():
        errors.append("Missing HTML output or index.html")
    if not (root / ".nojekyll").is_file():
        errors.append("Missing .nojekyll for portable Pages output")
    # Each locale must be a complete site with its own index and no source dump.
    localized = {}
    for locale in ('zh', 'en'):
        folder = root / locale
        localized[locale] = {p.relative_to(folder) for p in folder.rglob('*.html')
                             if '_static' not in p.relative_to(folder).parts}
        for required in ('index.html', 'search.html', 'genindex.html', 'searchindex.js'):
            if not (folder / required).is_file():
                errors.append(f'Missing {locale} output: {required}')
        if (folder / '_sources').exists() and any((folder / '_sources').rglob('*')):
            errors.append(f'Unexpected {locale} published source directory')
        for relative in localized[locale]:
            path = folder / relative
            content = path.read_text(encoding='utf-8')
            language = 'zh-CN' if locale == 'zh' else 'en'
            if not re.search(r'<html\b[^>]*\blang="' + language + r'"', content):
                errors.append(f'Wrong document language: {locale}/{relative}')
            if 'data-sw-language="zh"' not in content or 'data-sw-language="en"' not in content:
                errors.append(f'Missing language switch: {locale}/{relative}')
            if locale == 'en' and re.search(r'[\u4e00-\u9fff]', re.sub(r'<a [^>]*lang="zh-CN"[^>]*>中文</a>', '', content)):
                errors.append(f'Untranslated Chinese in English HTML: {relative}')
    if localized['zh'] != localized['en']:
        errors.append('Localized HTML page coverage differs')
    for relative in localized['zh'] & localized['en']:
        zh = pages[root / 'zh' / relative]
        en = pages[root / 'en' / relative]
        if {x for x in zh.ids if x.startswith('sw-section-')} != {x for x in en.ids if x.startswith('sw-section-')}:
            errors.append(f'Localized section anchors differ: {relative}')
        if {x for x in zh.ids if x.startswith('starwave.')} != {x for x in en.ids if x.startswith('starwave.')}:
            errors.append(f'Localized API anchors differ: {relative}')
        redirect = root / relative
        if not redirect.is_file() or 'location.replace' not in redirect.read_text(encoding='utf-8'):
            errors.append(f'Missing compatibility route: {relative}')
    api_redirects = json.loads((project / 'tools' / 'api_redirects.json').read_text())
    for source, routes in api_redirects.items():
        locale = 'zh' if source == 'root' else source
        for relative, route in routes.items():
            canonical = pages[root / locale / route.get('target_page', 'usage.html')]
            path = root / relative if source == 'root' else root / source / relative
            content = path.read_text(encoding='utf-8')
            if 'location.replace' not in content or 'noindex, follow' not in content:
                errors.append(f'Missing noindex API redirect: {source}/{relative}')
            for target in {route['default'], *route['fragments'].values()}:
                if target not in canonical.ids:
                    errors.append(f'Invalid legacy API target: {source}/{relative} -> {target}')
    contracts = json.loads((project / 'tools' / 'api_contract.json').read_text())
    modeling_redirects = json.loads((project / 'tools' / 'modeling_redirects.json').read_text())
    for source, routes in modeling_redirects.items():
        locale = 'zh' if source == 'root' else source
        canonical = pages[root / locale / 'modeling/wave-propagation.html']
        for relative, route in routes.items():
            path = root / relative if source == 'root' else root / source / relative
            content = path.read_text(encoding='utf-8')
            if 'location.replace' not in content or 'noindex, follow' not in content or 'wave-propagation.html' not in content:
                errors.append(f'Missing canonical modeling redirect: {source}/{relative}')
            for target in {route['default'], *route['fragments'].values()}:
                if target not in canonical.ids:
                    errors.append(f'Invalid legacy modeling target: {source}/{relative} -> {target}')
    for locale in ('zh', 'en'):
        usage = root / locale / 'usage.html'
        content = usage.read_text(encoding='utf-8')
        for name, contract in contracts.items():
            api_page = root / locale / (contract.get('page', 'usage') + '.html')
            if api_page not in pages or f'starwave.{name}' not in pages[api_page].ids:
                errors.append(f'Missing public API anchor: {locale}/{name}')
        if not re.search(r'<h1>Usage<a', content):
            errors.append(f'Wrong Usage title: {locale}')
        if 'Usage / API' in content or 'toctree-l2' in content:
            errors.append(f'Navigation is not flat: {locale}')
        anchors = re.findall(r'data-sw-anchor="([^"]+)"', content)
        if tuple(anchors) != USAGE_SECTIONS or len(set(anchors)) != len(USAGE_SECTIONS) or any(anchor not in pages[usage].ids for anchor in anchors):
            errors.append(f'Missing or invalid stable Usage section metadata: {locale}')
        for anchor, number in zip(USAGE_SECTIONS, USAGE_NUMBERS):
            if pages[usage].id_sections.get(anchor) != f'sw-section-{number}':
                errors.append(f'Usage positional anchor drift: {locale}/{anchor}')
        aliases = json.loads((project / 'tools' / 'usage_anchors.json').read_text())[locale]
        for alias, target in aliases.items():
            if alias not in pages[usage].ids or pages[usage].id_sections.get(alias) != pages[usage].id_sections.get(target):
                errors.append(f'Legacy Usage anchor drift: {locale}/{alias} -> {target}')
        for name, label in (('scalar', 'Scalar Function'), ('vrz', 'VRZ Function'),
                            ('vti', 'VTI Function'), ('elastic', 'Elastic Function')):
            if content.count(f'<h2>{label}<a') != 1:
                errors.append(f'Missing or repeated Usage function heading: {locale}/{name}')
        if re.search(r'<h3>(?:Function|函数)<a', content):
            errors.append(f'Redundant Usage function subheading: {locale}')
        other = pages[root / ('en' if locale == 'zh' else 'zh') / 'usage.html']
        if any(anchor not in other.ids for anchor in anchors):
            errors.append(f'Usage section cannot switch language: {locale}')
        index = (root / locale / 'searchindex.js').read_text(encoding='utf-8')
        docnames = json.loads(index.removeprefix('Search.setIndex(').removesuffix(')'))['docnames']
        if not {'usage', 'visco-gsls'} <= set(docnames) or 'visco-sls' in docnames or any(name.startswith('api/') for name in docnames):
            errors.append(f'API search entries are duplicated or missing: {locale}')
        if not {'modeling/wave-propagation', 'modeling/acquisition', 'modeling/reconstruction', 'docker', 'presentation'} <= set(docnames) or {'modeling/vrz', 'modeling/vti'} & set(docnames):
            errors.append(f'Modeling search entries are duplicated or missing: {locale}')
        combined = (root / locale / 'modeling/wave-propagation.html').read_text()
        if len(re.findall(r'data-sw-anchor="wave-[^"]+"', combined)) != 7:
            errors.append(f'Incomplete combined modeling sections: {locale}')
    links = 0
    for path, page in pages.items():
        if page.duplicate_ids:
            errors.append(f'Duplicate IDs: {path.relative_to(root)}: {sorted(page.duplicate_ids)}')
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            links += 1
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(root):
                errors.append(f"Escaping link: {path.name}: {href}")
            elif not target.exists():
                errors.append(f"Missing target: {path.name}: {href}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f"Missing anchor: {path.name}: {href}")
    figure_dir = project / 'docs' / '_static' / 'tutorials'
    figure_manifest = json.loads((figure_dir / 'figure_manifest.json').read_text())
    listed = set()
    for figure in figure_manifest['figures']:
        filename = figure['asset']
        listed.add(filename)
        path = figure_dir / filename
        data = path.read_bytes()
        if path.parent != figure_dir or not filename.endswith('.png'):
            errors.append(f'Invalid scientific asset: {filename}')
        if data[:8] != b'\x89PNG\r\n\x1a\n' or hashlib.sha256(data).hexdigest() != figure['asset_sha256']:
            errors.append(f'Scientific figure signature/hash mismatch: {filename}')
        elif struct.unpack('>II', data[16:24]) != (figure['width_px'], figure['height_px']):
            errors.append(f'Scientific figure dimensions mismatch: {filename}')
        if not figure.get('caption') or not figure.get('sources') or not figure.get('units'):
            errors.append(f'Missing scientific figure evidence: {filename}')
    if listed != {path.name for path in figure_dir.glob('*.png')} or len(listed) != 12:
        errors.append('Curated scientific figure set differs from manifest')
    for locale in ('zh', 'en'):
        for chapter in ('installation', 'modeling/gradient', 'inversion/fwi'):
            content = (root / locale / (chapter + '.html')).read_text()
            if '0.1.0.dev9' not in content or '.ipynb' not in content:
                errors.append(f'Missing tutorial scope/download: {locale}/{chapter}')
        fwi = pages[root / locale / 'inversion/fwi.html']
        aliases = {'id1': 'sw-section-1', 'id2': 'sw-section-2', 'id3': 'sw-section-3'} if locale == 'zh' else {
            'run-one-update': 'sw-section-1', 'key-points-for-the-loop': 'sw-section-2', 'validation-limits': 'sw-section-3'}
        for alias, target in aliases.items():
            if fwi.id_sections.get(alias) != target:
                errors.append(f'Legacy FWI anchor drift: {locale}/{alias}')

    # Only the selected, unchanged public logo assets may enter the site.
    brand_dir = project / 'docs' / '_static' / 'brand'
    brand_manifest = json.loads((brand_dir / 'manifest.json').read_text())
    brand_names = {'starwave-main.svg', 'starwave-icon-white.svg', 'starwave-favicon.png'}
    if {asset['file'] for asset in brand_manifest['assets']} != brand_names:
        errors.append('Unexpected brand manifest entries')
    if {p.name for p in brand_dir.iterdir()} != brand_names | {'manifest.json'}:
        errors.append('Unexpected brand files')
    for asset in brand_manifest['assets']:
        path = brand_dir / asset['file']
        if path.name not in brand_names or path.parent != brand_dir:
            errors.append('Invalid brand asset path')
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != asset['sha256']:
            errors.append(f'Brand asset changed: {path.name}')
        if path.suffix == '.svg':
            try:
                svg = ET.fromstring(data)
                for element in svg.iter():
                    tag = element.tag.rsplit('}', 1)[-1]
                    if tag in {'script', 'foreignObject', 'image', 'use', 'style'}:
                        errors.append(f'Unsafe SVG element: {path.name}/{tag}')
                    for key, value in element.attrib.items():
                        if key.lower().startswith('on') or ('href' in key.lower() and not value.startswith('#')):
                            errors.append(f'External or active SVG attribute: {path.name}/{key}')
                        if 'url(' in value and not re.fullmatch(r'url\(#[A-Za-z0-9_-]+\)', value):
                            errors.append(f'External SVG reference: {path.name}/{key}')
            except ET.ParseError:
                errors.append(f'Invalid brand SVG: {path.name}')
        elif data[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', data[16:24]) != (32, 32):
            errors.append('Favicon is not the selected 32px PNG')
    for locale in ('zh', 'en'):
        homepage = (root / locale / 'index.html').read_text()
        if 'sw-api-overview' in homepage or '<table' in homepage:
            errors.append(f'Removed homepage comparison returned: {locale}')
        if not all(x in homepage for x in ['sw-home-hero', 'sw-home-logo', 'sw-home-capabilities', 'sw-home-code', 'sw-home-paths']):
            errors.append(f'Incomplete product homepage: {locale}')
        if 'sw-sidebar-brand' not in homepage or 'starwave-favicon.png' not in homepage:
            errors.append(f'Missing brand navigation or favicon: {locale}')
        for name in brand_names:
            if (root / locale / '_static' / 'brand' / name).read_bytes() != (brand_dir / name).read_bytes():
                errors.append(f'Published brand asset differs: {locale}/{name}')

    diagram = project / 'docs' / '_static' / 'diagrams' / 'acquisition.svg'
    try:
        svg = ET.fromstring(diagram.read_bytes())
        if svg.tag.rsplit('}', 1)[-1] != 'svg' or not svg.get('viewBox'):
            errors.append('Invalid acquisition diagram root or dimensions')
        if not any(e.tag.rsplit('}', 1)[-1] == 'title' and e.text for e in svg.iter()):
            errors.append('Missing accessible acquisition diagram title')
        for element in svg.iter():
            if element.tag.rsplit('}', 1)[-1] in {'script', 'foreignObject', 'image', 'use'}:
                errors.append('Unsafe acquisition diagram element')
            for key, value in element.attrib.items():
                if key.lower().startswith('on') or ('href' in key.lower() and not value.startswith('#')):
                    errors.append('External or active acquisition diagram attribute')
                if 'url(' in value and not re.fullmatch(r'url\(#[A-Za-z0-9_-]+\)', value):
                    errors.append('External acquisition diagram reference')
        for locale in ('zh', 'en'):
            if (root / locale / '_static' / 'diagrams' / diagram.name).read_bytes() != diagram.read_bytes():
                errors.append(f'Published acquisition diagram differs: {locale}')
    except (FileNotFoundError, ET.ParseError):
        errors.append('Missing or invalid acquisition diagram')

    font_dir = project / 'docs' / '_static' / 'fonts'
    font_manifest = json.loads((font_dir / 'manifest.json').read_text())
    font_names = {'home-sans-sc-regular.woff', 'home-sans-sc-bold.woff'}
    if {p.name for p in font_dir.iterdir()} != font_names | {'OFL.txt', 'manifest.json'}:
        errors.append('Unexpected homepage font files')
    # Hidden toctrees supply the sidebar, which retains RTD's own typeface.
    # Only the homepage/Usage article text uses the custom Chinese glyph subset.
    home_source = re.sub(r'```\{toctree\}.*?\n```', '',
                         (project / 'docs' / 'index.md').read_text(), flags=re.S)
    home_source += (project / 'docs' / 'usage.md').read_text()
    chinese = set(re.findall(r'[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]', home_source))
    for font in font_manifest['assets']:
        path = font_dir / font['file']
        data = path.read_bytes()
        if path.name not in font_names or data[:4] != b'wOFF' or len(data) > 300000:
            errors.append('Invalid or oversized homepage font subset')
        if hashlib.sha256(data).hexdigest() != font['sha256'] or not chinese <= set(font['characters']):
            errors.append('Homepage font subset needs regeneration')
        for locale in ('zh', 'en'):
            if (root / locale / '_static' / 'fonts' / path.name).read_bytes() != data:
                errors.append(f'Published font differs: {locale}/{path.name}')
    license_text = (font_dir / 'OFL.txt').read_text()
    if 'SIL OPEN FONT LICENSE Version 1.1' not in license_text or 'Adobe' not in license_text:
        errors.append('Missing font license or copyright')

    # These checks are a guardrail, not a substitute for human content review.
    forbidden = [
        re.compile(r"[A-Za-z]:[\\/](?:Users|home)[\\/]", re.I),
        re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
        re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
        re.compile(r"(?:RELEASE_VERIFICATION|MAINTAINER_HANDOFF|HESS_JOINT_VP_WATER_FIX)"),
    ]
    allowed = {".md", ".py", ".txt", ".css", ".yml", ".yaml", ".html", ".js", ".json"}
    scalar3d_assets = approved_assets()
    sources = []
    for p in project.rglob("*"):
        relative = p.relative_to(project)
        if any(part in {".git", ".venv", "_build", "_readthedocs", "__pycache__"} for part in relative.parts):
            continue
        if not p.is_file():
            continue
        sources.append(p)
        is_tutorial_png = p.suffix == ".png" and relative.parts[:3] == ("docs", "_static", "tutorials")
        is_brand_file = relative.parts[:3] == ("docs", "_static", "brand") and p.name in brand_names
        is_font_file = relative.parts[:3] == ("docs", "_static", "fonts") and p.name in font_names
        is_diagram_file = relative.as_posix() == 'docs/_static/diagrams/acquisition.svg'
        is_reconstruction_file = relative.parts[:3] == ('docs', '_static', 'reconstruction') and p.name in RECONSTRUCTION_ASSETS
        is_tutorial_notebook = p.suffix == ".ipynb" and relative.parts[:2] == ("examples", "tutorials")
        is_presentation = p == ASSET_PATH
        is_scalar3d_asset = p in scalar3d_assets
        if p.name not in {".gitignore", ".gitattributes"} and p.suffix not in allowed and not is_tutorial_png and not is_tutorial_notebook and not is_brand_file and not is_font_file and not is_diagram_file and not is_reconstruction_file and not is_presentation and not is_scalar3d_asset:
            errors.append(f"Unexpected source file: {relative}")
        if p.suffix == ".py":
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                errors.append(f"Python syntax: {relative}: {exc.msg}")
        if p not in {Path(__file__).resolve(), project / 'tools' / 'check_presentation_asset.py', project / 'tools' / 'check_sls_examples.py'} and not is_tutorial_png and not (is_brand_file and p.suffix == ".png") and not is_font_file and not is_presentation and not is_scalar3d_asset:
            text = p.read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in forbidden):
                errors.append(f"Review restricted content: {relative}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in {".html", ".js", ".txt", ".py", ".css", ".ipynb", ".json"}:
            if any(pattern.search(p.read_text(encoding="utf-8")) for pattern in forbidden):
                errors.append(f"Review generated content: {p.relative_to(root)}")
    if (root / "_sources").exists() and any((root / "_sources").rglob("*")):
        errors.append("Unexpected published source directory")
    if (root / ".doctrees").exists():
        errors.append("Build cache inside publication directory; use -d _build/doctrees")
    if errors:
        print("\n".join(sorted(set(errors))))
        return 1
    print(f"PASS: {len(pages)} HTML pages, {links} local links/resources, {len(sources)} source files; syntax/content checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check(Path(sys.argv[1] if len(sys.argv) > 1 else "_build/html")))
