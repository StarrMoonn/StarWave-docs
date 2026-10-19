"""Verify the byte-preserved CPU wavefield assets and bilingual examples."""
from pathlib import Path
import hashlib
import json
import re
import struct
import sys
import zipfile
from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / 'docs/_static/wavefield2d'
FAMILIES = ('acoustic', 'elastic', 'vti', 'vrz', 'vrr', 'gsls')
TIMES = ('0200', '0350', '0500')


def manifest():
    return json.loads((ASSETS / 'manifest.json').read_text())


def approved_assets():
    return {ASSETS / entry['path'] for entry in manifest()['assets']} | {ASSETS / 'manifest.json'}


def check(root):
    from check_site import HTMLTree, element_text
    root = Path(root).resolve()
    errors = []
    m = manifest()
    if m['source_commit'] != '391f4a45efdee2f88a9d1432997fdca20ecd2d6f' or m['cuda_executed'] is not False:
        errors.append('Wavefield2D CPU provenance changed')
    actual = {p for p in ASSETS.rglob('*') if p.is_file()}
    if actual != approved_assets():
        errors.append('Wavefield2D strict asset allowlist differs')
    for entry in m['assets']:
        p = ASSETS / entry['path']
        if not p.is_file() or not p.resolve().is_relative_to(ASSETS.resolve()):
            errors.append('Missing/escaping Wavefield2D asset: ' + entry['path'])
            continue
        b = p.read_bytes()
        if len(b) != entry['bytes'] or hashlib.sha256(b).hexdigest() != entry['sha256']:
            errors.append('Wavefield2D hash mismatch: ' + entry['path'])
        if p.suffix == '.png':
            if b[:8] != b'\x89PNG\r\n\x1a\n' or list(struct.unpack('>II', b[16:24])) != entry['pixels']:
                errors.append('Wavefield2D PNG shape: ' + entry['path'])
        if p.suffix == '.npz':
            with zipfile.ZipFile(p) as archive:
                if archive.testzip() or not {'snapshots.npy', 'x_m.npy', 'z_m.npy'} <= set(archive.namelist()):
                    errors.append('Wavefield2D invalid snapshot archive: ' + entry['path'])
        if p.suffix == '.pdf':
            reader = PdfReader(p, strict=True)
            if reader.is_encrypted or len(reader.pages) != 1 or reader.attachments:
                errors.append('Wavefield2D unexpected PDF structure: ' + entry['path'])
            from check_presentation_asset import FORBIDDEN_KEYS, RESTRICTED_TEXT
            seen = set()
            def visit(value):
                if isinstance(value, IndirectObject):
                    identity = (value.idnum, value.generation)
                    if identity in seen:
                        return
                    seen.add(identity)
                    visit(value.get_object())
                elif isinstance(value, DictionaryObject):
                    for key, child in value.items():
                        if key in FORBIDDEN_KEYS or key == '/A':
                            errors.append('Wavefield2D active PDF feature: ' + entry['path'] + '/' + key)
                        visit(child)
                elif isinstance(value, ArrayObject):
                    for child in value:
                        visit(child)
            visit(reader.trailer)
            if RESTRICTED_TEXT.search(str(reader.metadata)) or any(RESTRICTED_TEXT.search(page.extract_text()) for page in reader.pages):
                errors.append('Wavefield2D restricted PDF content: ' + entry['path'])
        if p.suffix == '.svg' and re.search(r'<script|<foreignObject|\bon\w+\s*=|javascript:', b.decode(), re.I):
            errors.append('Wavefield2D active SVG content: ' + entry['path'])
        for locale in ('zh', 'en'):
            published = root / locale / '_static/wavefield2d' / entry['path']
            if not published.is_file() or published.read_bytes() != b:
                errors.append('Published Wavefield2D asset differs: ' + locale + '/' + entry['path'])
    headings = []
    for locale in ('zh', 'en'):
        page = root / locale / 'examples/wavefield-snapshots.html'
        if not page.is_file():
            errors.append('Missing Wavefield2D page: ' + locale)
            continue
        tree = HTMLTree(page.read_text()).root
        main = tree.find('.//*[@role="main"]')
        if main is None:
            errors.append('Missing Wavefield2D article')
            continue
        text = element_text(main)
        for term in ('391f4a45', 'CPU', 'CUDA', 'GSLS', 'VTI', 'VRZ', 'VRR', '0.20', '0.35', '0.50', '10.1190'):
            if term not in text:
                errors.append('Missing Wavefield2D evidence term: ' + locale + '/' + term)
        imgs = list(main.iter('img'))
        expected = {f'../_static/wavefield2d/panels/{family}_t{time}ms.png' for family in FAMILIES for time in TIMES}
        if not expected <= {i.get('src') for i in imgs}:
            errors.append('Missing six-family, three-time Wavefield2D panels: ' + locale)
        for i in imgs:
            if len(i.get('alt', '')) < 12:
                errors.append('Missing Wavefield2D figure alternative: ' + locale)
        links = {a.get('href', '') for a in main.iter('a')}
        for family in FAMILIES:
            if f'../_static/wavefield2d/data/{family}_snapshots.npz' not in links:
                errors.append('Missing raw Wavefield2D download: ' + locale + '/' + family)
        sections = list(main.iter('section'))
        headings.append(len(sections))
        if not any('sw-example-page' in s.get('class', '').split() for s in sections):
            errors.append('Missing Wavefield2D example style')
        if 'examples/wavefield-snapshots.html' not in (root / locale / 'index.html').read_text():
            errors.append('Missing Wavefield2D navigation: ' + locale)
    if len(set(headings)) != 1:
        errors.append('Wavefield2D bilingual heading parity differs')
    return errors


if __name__ == '__main__':
    errors = check(sys.argv[1] if len(sys.argv) > 1 else PROJECT / '_build/html')
    if errors:
        print('\n'.join(sorted(set(errors))))
        raise SystemExit(1)
    print('PASS: Wavefield2D original CPU assets, strict hashes, six families, three times, bilingual pages and downloads.')
