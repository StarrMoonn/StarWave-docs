"""Validate the bounded 2-D CPU free-surface publication, not PDE accuracy."""
from array import array
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import zipfile
from urllib.parse import unquote, urlsplit

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / 'docs/_static/free-surface'
SOURCE_COMMIT = '200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb'
CASES = ('scalar', 'born', 'vrz', 'vrr', 'vti', 'tti', 'elastic',
         'elastic_vrr', 'elastic_tti', 'gsls')
VALIDATIONS = ('scalar_mirror',)
ASSET_NAMES = {f'{name}.{suffix}' for name in CASES + VALIDATIONS
               for suffix in ('png', 'npz', 'json')} | {'born_background.png'}


def manifest():
    return json.loads((ASSETS / 'manifest.json').read_text(encoding='utf-8'))


def approved_assets():
    # This set is deliberately independent of untrusted manifest paths.
    return {ASSETS / name for name in ASSET_NAMES | {'manifest.json'}}


def numeric_npy(data, *, coordinate=False):
    """Read small numerical NPY records without NumPy or pickle execution."""
    stream = io.BytesIO(data)
    if stream.read(6) != b'\x93NUMPY':
        raise ValueError('invalid NPY magic')
    major, minor = stream.read(2)
    if (major, minor) not in {(1, 0), (2, 0), (3, 0)}:
        raise ValueError('unsupported NPY version')
    n = struct.unpack('<H' if major == 1 else '<I', stream.read(2 if major == 1 else 4))[0]
    if n > 65536:
        raise ValueError('excessive NPY header')
    h = ast.literal_eval(stream.read(n).decode('utf-8' if major == 3 else 'latin1').strip())
    shape, descr = h['shape'], h['descr']
    if not isinstance(shape, tuple) or any(type(x) is not int or x < 0 for x in shape):
        raise ValueError('invalid numerical array shape')
    allowed = {'<f4', '<f8', '>f4', '>f8', '|f4', '|f8'}
    if coordinate:
        allowed |= {'<i4', '<i8', '>i4', '>i8'}
    if h['fortran_order'] or descr not in allowed:
        raise ValueError('records must be C-order float32/float64; coordinates may also be int32/int64; objects are forbidden')
    size = int(descr[-1])
    payload = stream.read()
    if len(payload) != math.prod(shape) * size:
        raise ValueError('NPY byte count differs from shape')
    values = array(('f' if size == 4 else 'd') if descr[1] == 'f' else ('i' if size == 4 else 'q'))
    values.frombytes(payload)
    if (descr[0] == '>' and sys.byteorder == 'little') or (descr[0] == '<' and sys.byteorder == 'big'):
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError('nonfinite record or coordinate')
    return shape, values


def check_npz(path, name):
    errors = []
    lhs, rhs = ('free_surface', 'absorbing') if name in CASES else ('numerical', 'reference')
    required = (lhs, rhs, 'difference', 'time_s', 'receiver_x_m')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            return [f'Corrupt free-surface NPZ: {name}']
        if not {key + '.npy' for key in required} <= set(archive.namelist()):
            return [f'Missing free-surface raw-array keys: {name}']
        values = {key: numeric_npy(archive.read(key + '.npy'), coordinate=key in {'time_s', 'receiver_x_m'}) for key in required}
    shape = values[lhs][0]
    if len(shape) < 2 or values[rhs][0] != shape or values['difference'][0] != shape:
        errors.append(f'Free-surface gather shapes differ: {name}')
    elif values['time_s'][0] != (shape[-1],) or values['receiver_x_m'][0] != (shape[-2],):
        errors.append(f'Free-surface coordinates differ from gather: {name}')
    else:
        a, b, d = values[lhs][1], values[rhs][1], values['difference'][1]
        if not a or max(map(abs, a)) == 0 or max(map(abs, b)) == 0:
            errors.append(f'Empty/zero free-surface record: {name}')
        if any(abs(delta - (x - y)) > 2e-6 * max(abs(x), abs(y), 1e-30)
               for x, y, delta in zip(a, b, d)):
            errors.append(f'Free-surface difference is not raw subtraction: {name}')
        times = values['time_s'][1]
        if any(t1 <= t0 for t0, t1 in zip(times, times[1:])):
            errors.append(f'Non-increasing free-surface time axis: {name}')
    return errors


def check(root):
    from check_site import HTMLTree, element_text
    root = Path(root).resolve()
    errors = []
    try:
        m = manifest()
    except (OSError, ValueError) as exc:
        return [f'Missing/invalid free-surface manifest: {exc}']
    if (m.get('source_commit') != SOURCE_COMMIT or m.get('dimension') != 2
            or m.get('device') != 'cpu' or m.get('cuda_executed') is not False):
        errors.append('Free-surface source/2-D CPU provenance differs')
    cases = m.get('cases', [])
    if {c.get('id') for c in cases} != set(CASES) or len(cases) != len(CASES):
        errors.append('Free-surface manifest must describe exactly ten operators')
    if any(not c.get('observable') or not c.get('backend') or not c.get('dtype') for c in cases):
        errors.append('Missing per-operator component/backend/dtype')
    entries = m.get('assets', [])
    if {e.get('path') for e in entries} != ASSET_NAMES or len(entries) != len(ASSET_NAMES):
        errors.append('Free-surface manifest differs from the narrow asset allowlist')
    if {p for p in ASSETS.rglob('*') if p.is_file()} != approved_assets():
        errors.append('Unexpected/missing free-surface asset on disk')
    for entry in entries:
        name = entry.get('path')
        if name not in ASSET_NAMES:
            continue
        path = ASSETS / name
        if not path.is_file() or not path.resolve().is_relative_to(ASSETS.resolve()):
            errors.append('Missing/escaping free-surface asset: ' + name)
            continue
        data = path.read_bytes()
        if len(data) != entry.get('bytes') or hashlib.sha256(data).hexdigest() != entry.get('sha256'):
            errors.append('Free-surface asset hash mismatch: ' + name)
        try:
            if path.suffix == '.png':
                if (len(data) < 24 or data[:8] != b'\x89PNG\r\n\x1a\n'
                        or list(struct.unpack('>II', data[16:24])) != entry.get('pixels')):
                    errors.append('Free-surface PNG dimensions differ: ' + name)
            elif path.suffix == '.json':
                if not isinstance(json.loads(data), dict):
                    errors.append('Free-surface run metadata must be an object: ' + name)
            else:
                errors.extend(check_npz(path, path.stem))
        except (OSError, ValueError, KeyError, SyntaxError, struct.error, zipfile.BadZipFile) as exc:
            errors.append(f'Invalid free-surface asset {name}: {exc}')
        for locale in ('zh', 'en'):
            published = root / locale / '_static/free-surface' / name
            if not published.is_file() or published.read_bytes() != data:
                errors.append('Published free-surface asset differs: ' + locale + '/' + name)
    heading_counts = []
    for locale in ('zh', 'en'):
        page = root / locale / 'modeling/free-surface.html'
        if not page.is_file():
            errors.append('Missing free-surface chapter: ' + locale)
            continue
        rendered = page.read_text(encoding='utf-8')
        if 'INTEGRATION_REQUIRED' in rendered:
            errors.append('Unresolved free-surface draft marker: ' + locale)
        tree = HTMLTree(rendered).root
        main = tree.find('.//*[@role="main"]')
        if main is None:
            errors.append('Missing free-surface article: ' + locale)
            continue
        text = element_text(main)
        for term in ('200621cc31c0', 'CPU', 'CUDA', 'Born', 'GSLS', 'accuracy', 'coupling_order'):
            if term not in text:
                errors.append(f'Missing free-surface scope term: {locale}/{term}')
        images = list(main.iter('img'))
        expected_images = {f'../_static/free-surface/{name}.png' for name in CASES + VALIDATIONS}
        expected_images.add('../_static/free-surface/born_background.png')
        if {img.get('src') for img in images} != expected_images:
            errors.append('Missing/unexpected free-surface figure: ' + locale)
        if any(len(img.get('alt', '')) < 16 for img in images):
            errors.append('Missing free-surface image alternative: ' + locale)
        links = {a.get('href') for a in main.iter('a')}
        for name in CASES + VALIDATIONS:
            for suffix in ('npz', 'json'):
                if f'../_static/free-surface/{name}.{suffix}' not in links:
                    errors.append(f'Missing free-surface raw-data link: {locale}/{name}.{suffix}')
        sidebar = next((node for node in tree.iter('div')
                        if 'wy-menu-vertical' in node.get('class', '').split()), None)
        label = '自由表面' if locale == 'zh' else 'Free surface'
        targets = [] if sidebar is None else [a for a in sidebar.iter('a')
                    if not urlsplit(a.get('href', '')).scheme and not urlsplit(a.get('href', '')).netloc
                    and ((page.parent / unquote(urlsplit(a.get('href', '')).path)).resolve()
                         if urlsplit(a.get('href', '')).path else page) == page]
        if len(targets) != 1 or element_text(targets[0]) != label:
            errors.append('Missing/mislabeled free-surface sidebar entry: ' + locale)
        if sidebar is not None:
            group = None
            placed = False
            for child in sidebar:
                if child.tag == 'p' and 'caption' in child.get('class', '').split():
                    group = element_text(child)
                if child.tag == 'ul' and any(a in targets for a in child.iter('a')):
                    placed = group == ('正演模拟' if locale == 'zh' else 'Forward Modeling')
            if not placed:
                errors.append('Free surface is not under Forward Modeling: ' + locale)
        heading_counts.append(len(list(main.iter('section'))))
    if len(set(heading_counts)) > 1:
        errors.append('Free-surface bilingual section counts differ')
    return errors


if __name__ == '__main__':
    errors = check(sys.argv[1] if len(sys.argv) > 1 else PROJECT / '_build/html')
    if errors:
        print('\n'.join(sorted(set(errors))))
        raise SystemExit(1)
    print('PASS: Free-surface 2-D CPU provenance, strict assets, raw arrays, ten paired cases, scalar mirror, and bilingual navigation.')
