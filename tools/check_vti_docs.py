"""Check the current VTI-only overlay; preserve frozen release contracts."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ADDED = 'model_gradient_sampling_interval'


def parse(signature):
    return ast.parse('def documented' + signature + ': pass', feature_version=(3, 10)).body[0]


def untyped(args):
    args = copy.deepcopy(args)
    for node in ast.walk(args):
        if isinstance(node, ast.arg):
            node.annotation = None
    return ast.dump(args)


def check(source_root=None):
    snapshot = json.loads((ROOT / 'tools/vti_api_contract.json').read_text())
    assert snapshot['source_commit'] == '7145b5306af11615eb50c28a9ef64bb991a0bd97'
    assert set(snapshot['apis']) == {'vti'}, 'Only VTI may be overlaid'
    for file, key in [('api_contract.json', 'released_contract_sha256'),
                      ('v19_api_contract.json', 'v19_contract_sha256')]:
        assert hashlib.sha256((ROOT / 'tools' / file).read_bytes()).hexdigest() == snapshot[key], 'Historical snapshot changed: ' + file
    released = json.loads((ROOT / 'tools/api_contract.json').read_text())['vti']
    contract = snapshot['apis']['vti']
    expected, old = parse(contract['signature']), parse(released['signature'])
    assert len(expected.args.args + expected.args.kwonlyargs) == 25
    assert len(old.args.args + old.args.kwonlyargs) == 24
    trimmed = copy.deepcopy(expected.args)
    assert trimmed.kwonlyargs[-1].arg == ADDED
    assert ast.literal_eval(trimmed.kw_defaults[-1]) == 1
    trimmed.kwonlyargs = trimmed.kwonlyargs[:-1]
    trimmed.kw_defaults = trimmed.kw_defaults[:-1]
    assert untyped(trimmed) == untyped(old.args), 'Existing parameter/default changed'
    assert {k: v for k, v in contract['types'].items() if k != ADDED} == released['types']
    assert contract['types'][ADDED] == 'int'
    assert contract['return'] == released['return']
    for locale, folder in [('en', ROOT / 'docs/en'), ('zh', ROOT / 'docs')]:
        usage = (folder / 'usage.md').read_text()
        section = usage.split('(vti)=', 1)[1].split('(tti)=', 1)[0]
        matches = re.findall(r'^```\{py:function\} starwave.vti(\([^\n]+)\n(.*?)^```', section, re.M | re.S)
        assert len(matches) == 1
        signature, body = matches[0]
        assert ast.dump(parse(signature)) == ast.dump(expected), locale + ': signature drift'
        assert re.findall(r'^:param (\w+):', body, re.M) == [p.arg for p in expected.args.args + expected.args.kwonlyargs]
        assert dict(re.findall(r'^:type (\w+): `(.+)`$', body, re.M)) == contract['types']
        for marker in ('native_cpu', 'CPU/CUDA', 'full/boundary', 'full/checkpoint',
                       'eager/compile', '1..2^31-1', 'bool', 't % s == 0',
                       'min(s, internal_nt - t)', 'ceil(internal_nt/s)',
                       'rho = f(vp)', 'replicate-padding', 'CPML', 's=1',
                       snapshot['source_commit']):
            assert marker in section, f'{locale}: missing VTI scope marker {marker}'
        assert ('approximate' in section if locale == 'en' else '近似' in section)
        backend = (folder / 'pytorch-backend.md').read_text()
        example = backend.split('<!-- torch-api:vti:full:start -->', 1)[1].split('<!-- torch-api:vti:full:end -->', 1)[0]
        code = re.search(r'^```python\n(.*?)^```', example, re.M | re.S)[1]
        tree = ast.parse(code, feature_version=(3, 10))
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'starwave.vti')
        options = {kw.arg: kw.value for kw in call.keywords}
        assert set(options) == {p.arg for p in expected.args.kwonlyargs}
        assert ast.literal_eval(options[ADDED]) == 1
    if source_root:
        for file, sha256 in snapshot['source_files_sha256'].items():
            data = (Path(source_root) / file).read_bytes()
            assert hashlib.sha256(data).hexdigest() == sha256, 'Reviewed source mismatch: ' + file
        actual = next(n for n in ast.parse((Path(source_root) / 'starwave/vti.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'vti')
        assert untyped(actual.args) == untyped(expected.args)
        direct = next(n for n in ast.parse((Path(source_root) / 'starwave/torch_backend/vti.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'vti')
        options = dict(zip([p.arg for p in direct.args.kwonlyargs], direct.args.kw_defaults))
        assert ast.literal_eval(options[ADDED]) == 1
        assert ast.literal_eval(options['memory']) == 'full'
        print('PASS: exact reviewed VTI source hashes and public AST; direct Torch default full/sampling 1.')
    print('PASS: VTI 25-parameter bilingual overlay; immutable historical 24-parameter snapshots; sampling/storage scope.')
    return contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Reviewed VTI source tree; never imported')
    check(parser.parse_args().source_root)
