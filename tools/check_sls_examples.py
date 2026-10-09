"""Check the SLS public-example allowlist, downloads and bilingual evidence labels."""
from pathlib import Path
import ast
import hashlib
import io
import json
import re
import struct
import sys
import zipfile
from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / 'docs/_static/sls'
FIGURES = (
    'joint_models_initial', 'marmousi100_models_comparison',
    'marmousi100_loss_comparison', 'marmousi100_rmse_comparison',
    'joint_loss_and_model_errors', 'joint_Q_all_epochs',
    'joint_models_epoch_100', 'ring_Q_only_models', 'ring_joint_models', 'ring_loss',
)
PRIVATE = re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/|[A-Za-z]:[\\/](?:Users|home)[\\/]|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
TEXT = {'.py', '.md', '.txt', '.json', '.ipynb', '.html', '.log', '.js'}
DENIED_SUFFIXES = {'.c', '.cc', '.cpp', '.cu', '.h', '.hpp', '.so', '.dll', '.a', '.o', '.zip', '.gz', '.tar', '.pem'}


def manifest():
    return json.loads((ASSETS / 'manifest.json').read_text())


def approved_assets():
    return {ASSETS / entry['path'] for entry in manifest()['assets']} | {ASSETS / 'manifest.json'}


def inspect_public_text(data, name, errors):
    if Path(name).suffix not in TEXT:
        return
    text = data.decode('utf-8')
    if PRIVATE.search(text):
        errors.append('Private content in SLS public file: ' + name)
    if Path(name).suffix == '.py':
        try:
            ast.parse(text, feature_version=(3, 10))
        except SyntaxError:
            errors.append('SLS example Python 3.10 syntax: ' + name)


def check(root):
    from check_site import HTMLTree, element_text
    root = Path(root).resolve()
    errors = []
    m = manifest()
    if approved_assets() != {p for p in ASSETS.rglob('*') if p.is_file()}:
        errors.append('SLS asset allowlist differs')
    for entry in m['assets']:
        path = ASSETS / entry['path']
        if not path.resolve().is_relative_to(ASSETS.resolve()) or not path.is_file():
            errors.append('Missing/escaping SLS asset: ' + entry['path'])
            continue
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            errors.append('SLS asset integrity mismatch: ' + entry['path'])
        if path.suffix == '.png':
            if data[:8] != b'\x89PNG\r\n\x1a\n' or list(struct.unpack('>II', data[16:24])) != entry['pixels']:
                errors.append('SLS image dimensions: ' + entry['path'])
        inspect_public_text(data, entry['path'], errors)
        for locale in ('zh', 'en'):
            published = root / locale / '_static/sls' / entry['path']
            if not published.is_file() or published.read_bytes() != data:
                errors.append('Published SLS asset differs: ' + locale + '/' + entry['path'])
    package_path = ASSETS / m['package']['manifest']
    package = json.loads(package_path.read_text())
    parts = []
    for part in package['parts']:
        path = package_path.parent / part['file']
        if path.name != part['file']:
            errors.append('Escaping SLS package part')
            continue
        data = path.read_bytes()
        if len(data) > 4 * 1024 * 1024 or len(data) != part['bytes'] or hashlib.sha256(data).hexdigest() != part['sha256']:
            errors.append('SLS package part integrity: ' + part['file'])
        parts.append(data)
    archive_bytes = b''.join(parts)
    if len(archive_bytes) != package['bytes'] or len(archive_bytes) > 100 * 1024 * 1024 or hashlib.sha256(archive_bytes).hexdigest() != package['sha256']:
        errors.append('SLS complete package integrity/size')
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if archive.testzip():
            errors.append('SLS package CRC mismatch')
        prefix = m['package']['root'].rstrip('/') + '/'
        integrity = json.loads(archive.read(prefix + 'PACKAGE_MANIFEST.json'))['files']
        expected = {prefix + name for name in integrity} | {prefix + 'PACKAGE_MANIFEST.json', prefix + 'MANIFEST.sha256'}
        actual = {item.filename for item in archive.infolist() if not item.is_dir()}
        if expected != actual:
            errors.append('SLS internal package file allowlist differs')
        for line in archive.read(prefix + 'MANIFEST.sha256').decode('utf-8').splitlines():
            digest, name = line.split('  ', 1)
            if name.startswith('/') or '..' in Path(name).parts or hashlib.sha256(archive.read(prefix + name)).hexdigest() != digest:
                errors.append('SLS checksum manifest mismatch: ' + name)
        for name, record in integrity.items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or {'native', 'starwave', 'example_support', '.git', '__pycache__'} & set(path.parts) or path.suffix in DENIED_SUFFIXES:
                errors.append('Unapproved implementation/archive in SLS package: ' + name)
                continue
            data = archive.read(prefix + name)
            if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                errors.append('SLS package file integrity: ' + name)
            inspect_public_text(data, name, errors)
    pdf = PdfReader(ASSETS / 'downloads/SLS_Example_Report.pdf', strict=True)
    if len(pdf.pages) != 4 or pdf.is_encrypted or pdf.attachments:
        errors.append('Unexpected SLS PDF pages/encryption/attachments')
    for page in pdf.pages:
        if PRIVATE.search(page.extract_text()):
            errors.append('Private content in SLS PDF')
    from check_presentation_asset import FORBIDDEN_KEYS
    seen = set()
    def visit(value):
        if isinstance(value, IndirectObject):
            key = (value.idnum, value.generation)
            if key in seen:
                return
            seen.add(key)
            visit(value.get_object())
        elif isinstance(value, DictionaryObject):
            for key, child in value.items():
                if key in FORBIDDEN_KEYS:
                    errors.append('Unsafe SLS PDF capability: ' + key)
                if key == '/A' and child.get_object().get('/S') not in {'/URI', '/GoTo'}:
                    errors.append('Unsafe SLS PDF action')
                visit(child)
        elif isinstance(value, ArrayObject):
            for child in value:
                visit(child)
    visit(pdf.trailer)
    for locale in ('zh', 'en'):
        page = root / locale / 'examples/visco-sls.html'
        tree = HTMLTree(page.read_text()).root
        main = tree.find('.//*[@role="main"]')
        if main is None:
            errors.append('Missing SLS example article: ' + locale)
            continue
        figures = [f for f in main.iter('figure') if 'sw-example-figure' in f.get('class', '').split()]
        if len(figures) != len(FIGURES):
            errors.append('Incomplete SLS public figures: ' + locale)
        for figure, name in zip(figures, FIGURES):
            images = list(figure.iter('img'))
            expected = '../_static/sls/figures/' + name + '.png'
            if len(images) != 1 or images[0].get('src') != expected:
                errors.append('SLS figure source/order: ' + locale + '/' + name)
                continue
            if len(images[0].get('alt', '')) < 25 or len(element_text(figure.find('figcaption'))) < 40:
                errors.append('Missing SLS localized figure explanation')
            viewport = figure.find('div')
            if viewport is None or viewport.get('tabindex') != '0' or viewport.get('role') != 'region':
                errors.append('Missing SLS accessible figure scrolling')
        nav = next(e for e in tree.iter('div') if 'wy-menu-vertical' in e.get('class', '').split())
        captions = [element_text(e) for e in nav.iter('span') if 'caption-text' in e.get('class', '').split()]
        if captions.count('Example') != 1 or captions[captions.index('Example') + 1] != 'Usage':
            errors.append('SLS Example must immediately precede Usage')
        content = element_text(main)
        for value in ('70.131958', '82.819542', '59', '100', '0.003924164', '0.000950100'):
            if value not in content:
                errors.append('Missing SLS evidence distinction: ' + locale + '/' + value)
        buttons = [b for b in main.iter('button') if 'data-sw-package-download' in b.attrib]
        if len(buttons) != 1 or buttons[0].get('data-manifest') != '../_static/sls/downloads/package/package.json':
            errors.append('Missing SLS complete example download')
    return errors


if __name__ == '__main__':
    errors = check(sys.argv[1] if len(sys.argv) > 1 else PROJECT / '_build/html')
    if errors:
        print('\n'.join(sorted(set(errors))))
        raise SystemExit(1)
    print('PASS: SLS public asset/package allowlist, hashes, privacy scan, ten original figures, bilingual evidence labels and navigation.')
