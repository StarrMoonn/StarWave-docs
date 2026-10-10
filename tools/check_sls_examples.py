"""Enforce retirement of standalone SLS examples/assets from the current manual.

Original V14 evidence and integrity records remain in Git history. This gate
checks their absence from the publication, not their republication as an archive.
The old API URL is permitted only as a generated GSLS redirect.
"""
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
RETIRED_REFERENCES = ('examples/visco-sls.html', 'examples/visco-sls.md', '_static/sls/', 'SLS_Saved_Results.ipynb',
                      'SLS_Example_Report.pdf', 'marmousi2_sls_fwi.py')


def check(root):
    root = Path(root)
    errors = []
    for relative in ('docs/_static/sls', 'docs/examples/visco-sls.md',
                     'docs/en/examples/visco-sls.md', 'docs/visco-sls.md',
                     'docs/en/visco-sls.md'):
        if (PROJECT / relative).exists():
            errors.append('Retired SLS publication source remains: ' + relative)
    for path in (PROJECT / 'docs').rglob('*.md'):
        content = path.read_text(encoding='utf-8')
        if any(value in content for value in RETIRED_REFERENCES):
            errors.append('Retired SLS page/download reference: ' + str(path.relative_to(PROJECT)))
    for locale in ('', 'zh', 'en'):
        folder = root / locale
        for relative in ('examples/visco-sls.html', '_static/sls'):
            if (folder / relative).exists():
                errors.append('Retired SLS content remains published: ' + str((folder / relative).relative_to(root)))
        redirect = folder / 'visco-sls.html'
        if not redirect.is_file():
            errors.append('Missing old SLS API migration redirect: ' + locale)
        else:
            content = redirect.read_text(encoding='utf-8')
            if not all(marker in content for marker in ('location.replace', 'noindex, follow', 'visco-gsls.html')) or '<article' in content or 'py function' in content:
                errors.append('Old SLS API URL is not a redirect-only document: ' + locale)
    for path in root.rglob('*'):
        if path.is_file() and ('sls' in path.name.lower() and 'gsls' not in path.name.lower()) and path.name != 'visco-sls.html':
            errors.append('Retired SLS file remains in generated copies: ' + str(path.relative_to(root)))
    for path in root.rglob('*.html'):
        content = path.read_text(encoding='utf-8')
        if any(value in content for value in RETIRED_REFERENCES):
            errors.append('Published link to retired SLS content: ' + str(path.relative_to(root)))
        if re.search(r'id=[\"\']starwave\.(?:visco_sls|prepare_visco_sls|visco_sls_native_status)[\"\']', content):
            errors.append('Removed standalone SLS API definition remains: ' + str(path.relative_to(root)))
    return errors


if __name__ == '__main__':
    errors = check(sys.argv[1] if len(sys.argv) > 1 else PROJECT / '_build/html')
    if errors:
        print('\n'.join(errors))
        raise SystemExit(1)
    print('PASS: standalone SLS example pages, assets and download links are absent; old API URLs are GSLS redirects only.')
