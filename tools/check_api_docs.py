"""Validate bilingual public API docs without importing StarWave or PyTorch."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import zipfile

WHEEL_SHA256 = '6622b863c76db1ba708622048377f3de4447609ff7295d1705b8145884ce06db'
MODULES = {
    'scalar': 'starwave/scalar.py', 'vrz': 'starwave/vrz.py',
    'vti': 'starwave/vti.py', 'native_status': 'starwave/_native.py',
    'prepare_native': 'starwave/_native.py',
}


def parameter_defaults(args):
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    return list(zip(positional + args.kwonlyargs, defaults + args.kw_defaults))


def arguments_without_annotations(args):
    args = copy.deepcopy(args)
    for node in ast.walk(args):
        if isinstance(node, ast.arg):
            node.annotation = None
    return ast.dump(args)


def check(wheel_path=None):
    project = Path(__file__).resolve().parents[1]
    contracts = json.loads((project / 'tools' / 'api_contract.json').read_text())
    languages = {}
    blocks = 0
    for locale, docs in [('zh', project / 'docs'), ('en', project / 'docs' / 'en')]:
        functions = {}
        for path in sorted(docs.rglob('*.md')):
            if locale == 'zh' and 'en' in path.relative_to(docs).parts:
                continue
            text = path.read_text(encoding='utf-8')
            for code in re.findall(r'^```python\n(.*?)^```', text, re.M | re.S):
                ast.parse(code)
                blocks += 1
            for name, signature, body in re.findall(
                r'^```\{py:function\} starwave\.(\w+)(\([^\n]+|\(\)[^\n]*)\n(.*?)^```',
                text, re.M | re.S,
            ):
                assert name not in functions, f'{locale}: duplicate API directive: {name}'
                function = ast.parse(f'def {name}{signature}: pass').body[0]
                args = function.args
                expected = [p.arg for p, _ in parameter_defaults(args)]
                fields = re.findall(r'^:param (\w+):', body, re.M)
                types = dict(re.findall(r'^:type (\w+): `(.+)`$', body, re.M))
                assert fields == expected, f'{locale}.{name}: parameter descriptions/order mismatch'
                assert list(types) == expected, f'{locale}.{name}: parameter types/order mismatch'
                assert types == contracts[name]['types'], f'{locale}.{name}: public types changed'
                for param, default in parameter_defaults(args):
                    assert param.annotation is not None, f'{locale}.{name}.{param.arg}: missing annotation'
                    assert ast.dump(param.annotation) == ast.dump(ast.parse(types[param.arg], mode='eval').body)
                    description = re.search(r'^:param ' + param.arg + r': (.+)$', body, re.M)[1]
                    if locale == 'zh':
                        marker = '必填' if default is None else f'默认 `{ast.literal_eval(default)!r}`'
                    else:
                        marker = 'Required' if default is None else f'Default `{ast.literal_eval(default)!r}`'
                    assert marker in description, f'{locale}.{name}.{param.arg}: missing/wrong default'
                assert ':returns:' in body, f'{locale}.{name}: missing return contract'
                rtype = re.search(r'^:rtype: `(.+)`$', body, re.M)[1]
                assert rtype == contracts[name]['return']
                assert ast.dump(function.returns) == ast.dump(ast.parse(rtype, mode='eval').body)
                baseline = ast.parse(f"def {name}{contracts[name]['signature']}: pass").body[0].args
                assert arguments_without_annotations(args) == arguments_without_annotations(baseline), f'{locale}.{name}: public signature changed'
                functions[name] = args
        assert set(functions) == set(MODULES), f'{locale}: missing or unexpected API names'
        languages[locale] = functions
    for name in MODULES:
        assert ast.dump(languages['zh'][name]) == ast.dump(languages['en'][name]), f'{name}: language signature mismatch'
    if wheel_path:
        wheel_path = Path(wheel_path)
        assert hashlib.sha256(wheel_path.read_bytes()).hexdigest() == WHEEL_SHA256, 'Not the verified 2.0.0 wheel'
        with zipfile.ZipFile(wheel_path) as wheel:
            for name, module in MODULES.items():
                tree = ast.parse(wheel.read(module).decode('utf-8'))
                actual = next(n.args for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                assert arguments_without_annotations(languages['zh'][name]) == arguments_without_annotations(actual), f'{name}: differs from public wheel'
    counts = {name: len(parameter_defaults(languages['zh'][name])) for name in ('scalar', 'vrz', 'vti')}
    print(f'PASS: both languages: typed signatures, descriptions/defaults {counts}, 5 return contracts; {blocks} Python blocks parsed.')
    print('PASS: 5 signatures match verified public wheel.' if wheel_path else 'PASS: public contract snapshot; optional --wheel verifies the published archive.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', help='Public StarWave 2.0.0 wheel (read only, never imported)')
    check(parser.parse_args().wheel)
