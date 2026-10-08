"""Validate bilingual public API docs without importing StarWave or PyTorch."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import zipfile

# Public 4.0.0 artifact, independently matched to the publication receipt.
WHEEL_SHA256 = '9f64f2677c54af5b2bd1c48e509a24c7d40eb0257d9e86267e721d3f61eaa954'
MODULES = {
    'scalar': 'starwave/scalar.py', 'vrz': 'starwave/vrz.py',
    'vti': 'starwave/vti.py', 'native_status': 'starwave/_native.py',
    'prepare_native': 'starwave/_native.py',
    'elastic': 'starwave/elastic.py',
    'common.vpvsrho_to_lambmubuoyancy': 'starwave/common.py',
    'common.lambmubuoyancy_to_vpvsrho': 'starwave/common.py',
    'prepare_elastic': 'starwave/elastic_runtime.py',
}
WHEEL_TYPED_APIS = {
    'elastic', 'common.vpvsrho_to_lambmubuoyancy',
    'common.lambmubuoyancy_to_vpvsrho',
}


def parse_python(source, **kwargs):
    """All documented snippets and signatures must parse on Python 3.10."""
    return ast.parse(source, feature_version=(3, 10), **kwargs)


def parse_signature(signature):
    # A qualified public name (starwave.common.foo) is not a Python def name.
    return parse_python(f'def _documented_api{signature}: pass').body[0]


def wheel_function(wheel, module, name, seen=None):
    """Resolve direct definitions or import aliases without importing the wheel."""
    seen = set() if seen is None else seen
    key = (module, name)
    assert key not in seen, f'Circular wheel function alias: {module}:{name}'
    seen.add(key)
    tree = parse_python(wheel.read(module).decode('utf-8'))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) != name:
                    continue
                target = (node.module or '').split('.') if node.module else []
                if node.level:
                    package = module.removesuffix('.py').split('/')[:-1]
                    assert node.level <= len(package), f'Invalid relative import: {module}'
                    target = package[:len(package) - node.level + 1] + target
                imported = '/'.join(target)
                candidates = (imported + '.py', imported + '/__init__.py')
                source = next((path for path in candidates if path in wheel.namelist()), None)
                assert source, f'Unresolved wheel alias: {module}:{name}'
                return wheel_function(wheel, source, alias.name, seen)
    raise AssertionError(f'Missing public wheel function: {module}:{name}')


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
    assert set(contracts) == set(MODULES), 'Public contract inventory changed'
    languages = {}
    blocks = 0
    for locale, docs in [('zh', project / 'docs'), ('en', project / 'docs' / 'en')]:
        functions = {}
        for path in sorted(docs.rglob('*.md')):
            if locale == 'zh' and 'en' in path.relative_to(docs).parts:
                continue
            text = path.read_text(encoding='utf-8')
            for code in re.findall(r'^```python\n(.*?)^```', text, re.M | re.S):
                parse_python(code)
                blocks += 1
            for name, signature, body in re.findall(
                r'^```\{py:function\} starwave\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(\([^\n]+|\(\)[^\n]*)\n(.*?)^```',
                text, re.M | re.S,
            ):
                assert name not in functions, f'{locale}: duplicate API directive: {name}'
                assert name in contracts, f'{locale}: unexpected API directive: {name}'
                function = parse_signature(signature)
                args = function.args
                expected = [p.arg for p, _ in parameter_defaults(args)]
                fields = re.findall(r'^:param (\w+):', body, re.M)
                types = dict(re.findall(r'^:type (\w+): `(.+)`$', body, re.M))
                assert fields == expected, f'{locale}.{name}: parameter descriptions/order mismatch'
                assert list(types) == expected, f'{locale}.{name}: parameter types/order mismatch'
                assert types == contracts[name]['types'], f'{locale}.{name}: public types changed'
                for param, default in parameter_defaults(args):
                    assert param.annotation is not None, f'{locale}.{name}.{param.arg}: missing annotation'
                    assert ast.dump(param.annotation) == ast.dump(parse_python(types[param.arg], mode='eval').body), f'{locale}.{name}.{param.arg}: annotation differs from type field'
                    description = re.search(r'^:param ' + param.arg + r': (.+)$', body, re.M)[1]
                    if locale == 'zh':
                        marker = '必填' if default is None else f'默认 `{ast.literal_eval(default)!r}`'
                    else:
                        marker = 'Required' if default is None else f'Default `{ast.literal_eval(default)!r}`'
                    assert marker in description, f'{locale}.{name}.{param.arg}: missing/wrong default'
                assert ':returns:' in body, f'{locale}.{name}: missing return contract'
                rtype = re.search(r'^:rtype: `(.+)`$', body, re.M)[1]
                assert rtype == contracts[name]['return']
                assert ast.dump(function.returns) == ast.dump(parse_python(rtype, mode='eval').body)
                baseline = parse_signature(contracts[name]['signature']).args
                assert arguments_without_annotations(args) == arguments_without_annotations(baseline), f'{locale}.{name}: public signature changed'
                functions[name] = args
        assert set(functions) == set(MODULES), f'{locale}: missing or unexpected API names'
        languages[locale] = functions
    for name in MODULES:
        assert ast.dump(languages['zh'][name]) == ast.dump(languages['en'][name]), f'{name}: language signature mismatch'
    if wheel_path:
        wheel_path = Path(wheel_path)
        assert hashlib.sha256(wheel_path.read_bytes()).hexdigest() == WHEEL_SHA256, 'Not the verified 4.0.0 wheel'
        with zipfile.ZipFile(wheel_path) as wheel:
            for name, module in MODULES.items():
                actual = wheel_function(wheel, module, name.rsplit('.', 1)[-1])
                assert arguments_without_annotations(languages['zh'][name]) == arguments_without_annotations(actual.args), f'{name}: differs from public wheel'
                if name in WHEEL_TYPED_APIS:
                    actual_types = {param.arg: ast.unparse(param.annotation) for param, _ in parameter_defaults(actual.args)}
                    assert contracts[name]['types'] == actual_types, f'{name}: types differ from public wheel'
                    assert contracts[name]['return'] == ast.unparse(actual.returns), f'{name}: return type differs from public wheel'
    counts = {name: len(parameter_defaults(languages['zh'][name])) for name in MODULES}
    print(f'PASS: both languages: typed signatures, descriptions/defaults {counts}, {len(MODULES)} return contracts; {blocks} Python blocks parsed as Python 3.10.')
    print(f'PASS: {len(MODULES)} signatures match verified public 4.0.0 wheel.' if wheel_path else 'PASS: public 4.0.0 contract snapshot; optional --wheel verifies the published archive.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', help='Public StarWave 4.0.0 wheel (read only, never imported)')
    check(parser.parse_args().wheel)
