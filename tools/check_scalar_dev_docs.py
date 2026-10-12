"""Frozen V19 scalar contract; released API snapshots remain immutable."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ADDED = ('return_wavefield', 'forward_callback', 'callback_frequency')


def parse(signature):
    return ast.parse('def documented' + signature + ': pass', feature_version=(3, 10)).body[0]


def without_annotations(args):
    args = copy.deepcopy(args)
    for node in ast.walk(args):
        if isinstance(node, ast.arg):
            node.annotation = None
    return ast.dump(args)


def check(source_root=None):
    snapshot = json.loads((ROOT / 'tools/scalar_dev_api_contract.json').read_text())
    assert snapshot['source_commit'] == 'fba1f496b890f2dbaf3fbb3011d39f6d723cfea1'
    released_file = ROOT / 'tools/api_contract.json'
    assert hashlib.sha256(released_file.read_bytes()).hexdigest() == snapshot['released_contract_sha256'], 'Released API contract must not be silently refreshed'
    released = json.loads(released_file.read_text())['scalar']
    assert set(snapshot['apis']) == {'scalar'}, 'Dev override may change scalar only'
    contract = snapshot['apis']['scalar']
    expected = parse(contract['signature'])
    old = parse(released['signature'])
    assert len(expected.args.args + expected.args.kwonlyargs) == 23
    assert len(old.args.args + old.args.kwonlyargs) == 20
    trimmed = copy.deepcopy(expected.args)
    assert tuple(p.arg for p in trimmed.kwonlyargs[-3:]) == ADDED
    assert [ast.literal_eval(v) for v in trimmed.kw_defaults[-3:]] == [False, None, 1]
    trimmed.kwonlyargs = trimmed.kwonlyargs[:-3]
    trimmed.kw_defaults = trimmed.kw_defaults[:-3]
    assert without_annotations(trimmed) == without_annotations(old.args), 'Existing scalar arguments/defaults changed'
    assert {k:v for k,v in contract['types'].items() if k not in ADDED} == released['types']
    assert contract['return'] == 'tuple[torch.Tensor, ...]'
    semantics = contract['semantics']
    assert semantics['source_gradient_by_dimension'] == {'2': True, '3': True}
    assert semantics['all_zero_faces'] is True and semantics['positive_widths_equal'] is True
    assert semantics['callback_clock'] == 'completed user steps'
    assert semantics['callback_checkpoint_replay'] is False
    snippets = []
    for locale, folder in [('en', ROOT / 'docs/en'), ('zh', ROOT / 'docs')]:
        usage = (folder / 'usage.md').read_text()
        text = usage.split('(scalar)=', 1)[1].split('(born)=', 1)[0]
        matches = re.findall(r'^```\{py:function\} starwave.scalar(\([^\n]+)\n(.*?)^```', usage, re.M | re.S)
        assert len(matches) == 1
        signature, body = matches[0]
        assert ast.dump(parse(signature)) == ast.dump(expected), f'{locale}: scalar dev signature drift'
        assert re.findall(r'^:param (\w+):', body, re.M) == [p.arg for p in expected.args.args + expected.args.kwonlyargs]
        assert dict(re.findall(r'^:type (\w+): `(.+)`$', body, re.M)) == contract['types']
        for marker in ('20', 'V17', 'V18', 'V19', 'return_wavefield', 'forward_callback', 'callback_frequency',
                       'internal_step', 'internal_dt', 'state.time', '[B,0,D]',
                       'notebook/Scalar/Scalar2D_TargetWavefield_CUDA.ipynb',
                       'notebook/Scalar/Scalar2D_TargetWavefield_TorchCompile.py'):
            assert marker in text, f'{locale}: missing dev scope marker {marker}'
        assert 'notebook-dev/' not in text
        example = text.split('<!-- scalar-dev:example:start -->', 1)[1].split('<!-- scalar-dev:example:end -->', 1)[0]
        code = re.search(r'^```python\n(.*?)^```', example, re.M | re.S)[1]
        tree = ast.parse(code, feature_version=(3, 10))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'starwave.scalar']
        assert len(calls) == 1
        options = {kw.arg:kw.value for kw in calls[0].keywords}
        for name, value in {'return_wavefield': True, 'callback_frequency': 3,
                            'pml_width': 0, 'backend': 'torch', 'memory': 'checkpoint',
                            'checkpoint_interval': 4}.items():
            assert ast.literal_eval(options[name]) == value
        assert ast.unparse(options['forward_callback']) == 'observe'
        assert 'torch.empty((1, 0, 2)' in code
        assert 'v.grad, source.grad' in code and 'assert len(steps) == 3' in code
        snippets.append(code)
    assert snippets[0] == snippets[1], 'Bilingual dev scalar examples differ'
    if source_root:
        source = Path(source_root) / snapshot['source_file']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == json.loads((ROOT / 'tools/v19_source_hashes.json').read_text())[snapshot['source_file']], 'Scalar source differs from frozen V19'
        tree = ast.parse(source.read_text(), feature_version=(3, 10))
        actual = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scalar')
        assert without_annotations(actual.args) == without_annotations(expected.args)
        torch_tree = ast.parse((Path(source_root) / 'starwave/torch_backend/scalar.py').read_text(), feature_version=(3, 10))
        torch_api = next(n for n in torch_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scalar')
        index = [p.arg for p in torch_api.args.kwonlyargs].index('memory')
        assert ast.literal_eval(torch_api.args.kw_defaults[index]) == 'full'
        torch_api.args.kw_defaults[index] = ast.Constant(value='boundary')
        assert without_annotations(torch_api.args) == without_annotations(actual.args)
        print('PASS: V19 scalar and direct Torch ASTs match; only direct Torch memory default differs.')
    print('PASS: V19 scalar 23-parameter contract, immutable released 20-parameter baseline, callbacks and bilingual terminal-only example.')
    return contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Authorized current-dev source tree; never imported')
    check(parser.parse_args().source_root)
