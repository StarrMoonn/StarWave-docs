"""Check the independent Born public contract through AST, never solver imports."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67'
EXPECTED_COUNTS = {'scalar_born': 47, 'ScalarBorn': 13, 'ScalarBorn.forward': 41}


def parse(signature):
    return ast.parse('def public_api' + signature + ': pass', feature_version=(3, 10)).body[0]


def parameters(args):
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    return list(zip(positional + args.kwonlyargs, defaults + args.kw_defaults))


def check(source_root=None):
    contract = json.loads((ROOT / 'tools/born_api_contract.json').read_text())
    assert contract['source_commit'] == SOURCE_COMMIT
    assert contract['source_version'] == '0.1.0.dev17'
    assert contract['publication'] == {'source_release': 'V17', 'wheel_8_0_0': 'pending', 'published_wheel': '7.1.0'}
    assert {name: spec['parameter_count'] for name, spec in contract['apis'].items()} == EXPECTED_COUNTS
    snippets = []
    for locale, folder in [('zh', ROOT / 'docs'), ('en', ROOT / 'docs/en')]:
        usage = (folder / 'usage.md').read_text()
        assert usage.index('(scalar)=') < usage.index('(born)=') < usage.index('(vrz)=') < usage.index('(vrr)=')
        assert '- [Scalar Function](#scalar)\n- [Born Function](#born)\n- [VRZ Function](#vrz)' in usage
        text = usage.split('(born)=\n## Born Function', 1)[1].split('(vrz)=', 1)[0]
        for name, spec in contract['apis'].items():
            matches = re.findall(r'^```\{py:' + spec['kind'] + r'\} starwave\.' + re.escape(name) + r'(\([^\n]+)\n(.*?)^```', usage, re.M | re.S)
            assert len(matches) == 1, f'{locale}: missing/duplicated {name}'
            signature, body = matches[0]
            documented, pinned = parse(signature), parse(spec['signature'])
            assert ast.dump(documented) == ast.dump(pinned), f'{locale}/{name}: signature, annotation or default drift'
            fields = re.findall(r'^:param (\w+): (.+)$', body, re.M)
            expected = parameters(pinned.args)
            assert [name for name, _ in fields] == [arg.arg for arg, _ in expected], f'{locale}/{name}: parameter order/coverage'
            types = dict(re.findall(r'^:type (\w+): `(.+)`$', body, re.M))
            assert types == spec['types'], f'{locale}/{name}: parameter type drift'
            for ((arg, default), (_, description)) in zip(expected, fields):
                if default is None:
                    assert ('Required' if locale == 'en' else '必填') in description
                else:
                    value = ast.literal_eval(default)
                    assert (('Default' if locale == 'en' else '默认') + f' `{value!r}`') in description, f'{locale}/{name}/{arg.arg}: default prose'
                if arg.annotation is not None:
                    assert ast.dump(arg.annotation) == ast.dump(ast.parse(types[arg.arg], mode='eval').body)
            if spec['return'] is not None:
                assert ':returns:' in body and f":rtype: `{spec['return']}`" in body
        for section in ('backends', 'conventions', 'gradients', 'callbacks', 'returns', 'module', 'forward', 'example', 'build', 'scope'):
            assert f'(born-{section})=' in text, f'{locale}: missing {section}'
        for marker in ('δv', '1D/2D/3D', 'float32/64', 'MPS', 'AMP', 'CFL_ratio*k',
                       'state.step', 'state.dt', '10/14/18', 'out[-2]', 'out[-1]',
                       'compile_born.py --cpu-only', 'compile_all.py', 'V19'):
            assert marker in text, f'{locale}: missing scope marker {marker}'
        blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
        assert len(blocks) == 1, f'{locale}: exact standalone Born example required'
        example = ast.parse(blocks[0], feature_version=(3, 10))
        calls = [node for node in ast.walk(example) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'scalar_born']
        assert len(calls) == 1 and len(calls[0].args) == 4
        values = {kw.arg: kw.value for kw in calls[0].keywords}
        for name, value in {'backend': 'torch', 'memory': 'checkpoint', 'checkpoint_interval': 4,
                            'accuracy': 4, 'pml_width': 4, 'pml_freq': 20.0, 'max_vel': 1600.0}.items():
            assert ast.literal_eval(values[name]) == value, f'{locale}: CPU example {name}'
        assert {'source_amplitudes', 'source_locations', 'receiver_locations', 'bg_receiver_locations'} <= set(values)
        assert any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'loss.backward' for n in ast.walk(example))
        assert 'for x in (v, scatter, source)' in blocks[0] and 'torch.isfinite(x.grad).all()' in blocks[0]
        snippets.append(ast.dump(example))
        assert not (folder / 'born.md').exists(), 'Born belongs in Usage, not a duplicate page'
    assert snippets[0] == snippets[1], 'Bilingual Born executable examples drifted'
    if source_root:
        source = Path(source_root) / contract['source_file']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == json.loads((ROOT / 'tools/v19_source_hashes.json').read_text())[contract['source_file']], 'Born source differs from frozen V19 file'
        tree = ast.parse(source.read_text(), feature_version=(3, 10))
        module = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ScalarBorn')
        nodes = {'scalar_born': next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scalar_born'),
                 'ScalarBorn': next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'),
                 'ScalarBorn.forward': next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')}
        for name, node in nodes.items():
            args = copy.deepcopy(node.args)
            if name != 'scalar_born':
                assert args.args.pop(0).arg == 'self'
            pinned = parse(contract['apis'][name]['signature'])
            assert ast.dump(args) == ast.dump(pinned.args), f'{name}: actual V17 source signature drift'
            if name != 'ScalarBorn':
                assert ast.dump(node.returns) == ast.dump(pinned.returns)
        print('PASS: all three Born interfaces match the pinned source SHA-256 and source AST; no runtime imported.')
    print('PASS: bilingual Born function/module/forward: 47/13/41 parameters, defaults, types, return contracts, example and scope.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Optional authorized V17 source tree; AST-only read')
    check(parser.parse_args().source_root)
