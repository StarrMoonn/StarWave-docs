"""Check API parameter coverage; optionally compare a public 2.0.0 wheel by AST.

This does not import or execute StarWave and does not require PyTorch or CUDA.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import re
import zipfile

WHEEL_SHA256 = '6622b863c76db1ba708622048377f3de4447609ff7295d1705b8145884ce06db'
MODULES = {
    'scalar': 'starwave/scalar.py',
    'vrz': 'starwave/vrz.py',
    'vti': 'starwave/vti.py',
    'native_status': 'starwave/_native.py',
    'prepare_native': 'starwave/_native.py',
}


def parameter_defaults(args):
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    return list(zip(positional + args.kwonlyargs, defaults + args.kw_defaults))


def check(wheel_path=None):
    docs = Path(__file__).resolve().parents[1] / 'docs'
    functions = {}
    blocks = 0
    for path in sorted(docs.rglob('*.md')):
        text = path.read_text(encoding='utf-8')
        for code in re.findall(r'^```python\n(.*?)^```', text, re.M | re.S):
            ast.parse(code)
            blocks += 1
        for name, signature, body in re.findall(
            r'^```\{py:function\} starwave\.(\w+)(\([^\n]*\))\n(.*?)^```',
            text, re.M | re.S,
        ):
            assert name not in functions, f'Duplicate API directive: {name}'
            args = ast.parse(f'def {name}{signature}: pass').body[0].args
            expected = [p.arg for p, _ in parameter_defaults(args)]
            fields = re.findall(r'^:param (\w+):', body, re.M)
            types = re.findall(r'^:type (\w+):', body, re.M)
            assert fields == expected, f'{name}: parameter descriptions/order mismatch'
            assert types == expected, f'{name}: parameter types/order mismatch'
            for param, default in parameter_defaults(args):
                description = re.search(r'^:param ' + param.arg + r': (.+)$', body, re.M).group(1)
                marker = '必填' if default is None else f'默认 `{ast.literal_eval(default)!r}`'
                assert marker in description, f'{name}.{param.arg}: missing or wrong default'
            assert ':returns:' in body and ':rtype:' in body, f'{name}: missing return contract'
            functions[name] = args
    assert set(functions) == set(MODULES), 'Unexpected or missing documented API names'
    if wheel_path:
        wheel_path = Path(wheel_path)
        assert hashlib.sha256(wheel_path.read_bytes()).hexdigest() == WHEEL_SHA256, 'Not the verified 2.0.0 wheel'
        with zipfile.ZipFile(wheel_path) as wheel:
            for name, module in MODULES.items():
                tree = ast.parse(wheel.read(module).decode('utf-8'))
                actual = next(n.args for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                assert ast.dump(functions[name]) == ast.dump(actual), f'{name}: differs from public wheel signature'
    counts = {name: len(parameter_defaults(functions[name])) for name in ('scalar', 'vrz', 'vti')}
    print(f'PASS: parameter descriptions/types/defaults {counts}; 5 API return contracts; {blocks} Python blocks parsed.')
    print('PASS: 5 signatures match verified public wheel.' if wheel_path else 'Wheel comparison not requested; use --wheel for release verification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', help='Path to the public StarWave 2.0.0 wheel (read only)')
    check(parser.parse_args().wheel)
