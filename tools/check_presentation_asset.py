"""Validate the one reviewed, watermarked public presentation PDF.

The exact approved bytes are allowlisted; editable slides, source material,
attachments, active PDF actions and additional PDFs are never published.
These checks complement independent content and visual review.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

PROJECT = Path(__file__).resolve().parents[1]
ASSET_DIRECTORY = PROJECT / 'docs' / '_static' / 'presentations'
ASSET_NAME = 'starwave-presentation.pdf'
ASSET_PATH = ASSET_DIRECTORY / ASSET_NAME
FORBIDDEN_KEYS = {
    '/AA', '/OpenAction', '/JavaScript', '/JS', '/Launch', '/EmbeddedFiles',
    '/EF', '/RichMedia', '/RichMediaContent', '/Movie', '/Sound', '/XFA',
    '/AcroForm', '/Collection', '/SubmitForm', '/ImportData',
}
RESTRICTED_TEXT = re.compile(
    r'(?:[A-Za-z]:[\\/](?:Users|home)[\\/]|/(?:home|Users)/[A-Za-z0-9_.-]+/'
    r'|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'
    r'|RELEASE_VERIFICATION|MAINTAINER_HANDOFF|HESS_JOINT_VP_WATER_FIX)'
)


def check(root=None):
    errors = []
    try:
        manifest = json.loads((ASSET_DIRECTORY / 'manifest.json').read_text())
        assert manifest['file'] == ASSET_NAME
        assert manifest['pages'] == 30
        assert manifest['publication'] == 'original-watermarked-presentation'
        assert len(manifest['sha256']) == 64
        assert len(manifest['page_proofs']) == 30
        assert manifest['watermark']['visible_text']
        assert manifest['watermark']['method'] == 'baked-into-reviewed-page-image'
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        return [f'Invalid presentation allowlist: {exc}']
    if {p.name for p in ASSET_DIRECTORY.iterdir()} != {ASSET_NAME, 'manifest.json'}:
        errors.append('Unexpected presentation asset or authoring source')
    try:
        data = ASSET_PATH.read_bytes()
        if not data.startswith(b'%PDF-'):
            errors.append('Presentation is not a PDF')
        if hashlib.sha256(data).hexdigest() != manifest['sha256']:
            errors.append('Presentation differs from the reviewed SHA-256')
        if len(data) != manifest['bytes']:
            errors.append('Presentation size differs from its allowlist')
        reader = PdfReader(ASSET_PATH, strict=True)
        if reader.is_encrypted:
            return errors + ['Presentation is unexpectedly encrypted']
        if len(reader.pages) != 30:
            errors.append('Presentation must contain exactly 30 slides')
        for number, page in enumerate(reader.pages, 1):
            text = ' '.join(page.extract_text().split())
            proof = manifest['page_proofs'][number - 1]
            images = list(page['/Resources'].get('/XObject', {}).values())
            if len(images) != 1:
                errors.append(f'Expected one reviewed watermarked image on slide {number}')
            else:
                image = images[0].get_object()
                if (proof['page'] != number or image.get('/Subtype') != '/Image'
                        or image.get('/Width') != proof['width'] or image.get('/Height') != proof['height']
                        or hashlib.sha256(image.get_data()).hexdigest() != proof['image_sha256']):
                    errors.append(f'Reviewed watermarked pixels changed on slide {number}')
            if RESTRICTED_TEXT.search(text):
                errors.append(f'Restricted content on presentation slide {number}')
            box = page.mediabox
            if abs(float(box.width) / float(box.height) - 16 / 9) > 0.02:
                errors.append(f'Unexpected aspect ratio on slide {number}')
        if reader.attachments:
            errors.append('Presentation contains an embedded file')
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
                    if key in FORBIDDEN_KEYS:
                        errors.append(f'Unsafe or unexpected PDF capability: {key}')
                    if key == '/A':
                        action = child.get_object()
                        if not isinstance(action, DictionaryObject) or action.get('/S') not in {'/URI', '/GoTo'}:
                            errors.append('Unexpected PDF action')
                    if key == '/URI':
                        link = urlsplit(str(child))
                        if link.scheme not in {'https', 'http'} or not link.netloc:
                            errors.append('Unsafe PDF link target')
                    visit(child)
            elif isinstance(value, ArrayObject):
                for child in value:
                    visit(child)

        visit(reader.trailer)
        if RESTRICTED_TEXT.search(str(reader.metadata)):
            errors.append('Restricted presentation metadata')
    except Exception as exc:
        errors.append(f'Unable to validate the presentation PDF: {exc}')

    # No other deck/PDF may enter the source tree or either built locale.
    for path in PROJECT.rglob('*'):
        relative = path.relative_to(PROJECT)
        if any(part in {'.git', '.venv', '_build', '_readthedocs', '__pycache__'} for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in {'.pdf', '.ppt', '.pptx', '.odp', '.key'} and path != ASSET_PATH:
            errors.append(f'Unapproved presentation file: {relative}')
    if root is not None:
        root = Path(root).resolve()
        expected = {root / locale / '_static' / 'presentations' / ASSET_NAME for locale in ('zh', 'en')}
        actual = {p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.pdf', '.ppt', '.pptx', '.odp', '.key'}}
        if actual != expected:
            errors.append('Published presentation file set differs from the allowlist')
        for path in expected:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
                errors.append(f'Published presentation mismatch: {path.relative_to(root)}')
    return errors


if __name__ == '__main__':
    errors = check(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    if errors:
        raise SystemExit('\n'.join(sorted(set(errors))))
    print('PASS: approved 30-slide PDF, exact SHA-256, reviewed watermarked page-image hashes, safe structure, and publication allowlist.')
