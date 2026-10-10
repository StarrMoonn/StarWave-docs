"""Validate bilingual public API docs without importing StarWave or PyTorch."""
import argparse
import ast
import copy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import zipfile

# Public 7.0.0 wheel: official PyPI identity matches the reviewed CI artifact.
# Existing API/semantic entries retain their original review provenance.
WHEEL_VERSION = '7.0.0'
WHEEL_SHA256 = '0fb2d66555ad19d4178738db93ea5d71c3b7dbfb456cf4fd82898f5d44725964'
MODULES = {
    'scalar': 'starwave/scalar.py', 'vrz': 'starwave/vrz.py',
    'vti': 'starwave/vti.py', 'native_status': 'starwave/_native.py',
    'prepare_native': 'starwave/_native.py',
    'elastic': 'starwave/elastic.py',
    'common.vpvsrho_to_lambmubuoyancy': 'starwave/common.py',
    'common.lambmubuoyancy_to_vpvsrho': 'starwave/common.py',
    'prepare_elastic': 'starwave/elastic_runtime.py',
    'visco_gsls': 'starwave/visco_gsls.py',
    'prepare_visco_gsls': 'starwave/visco_gsls.py',
    'visco_gsls_native_status': 'starwave/visco_gsls.py',
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


def check_scalar3d_examples(project, contract):
    """Check runnable wiring against the reviewed semantic ledger, not numerics."""
    semantics = contract['semantics']
    assert semantics['reviewed_public_version'] == '5.0.0'
    assert semantics['model_dimensions'] == [2, 3]
    assert semantics['source_gradient_by_dimension'] == {'2': False, '3': True}
    assert semantics['pml_faces_by_dimension'] == {'2': 4, '3': 6}
    assert semantics['boundary_face_width_3d'] == 'accuracy // 2'
    assert semantics['boundary_terminal_pressure_fields_3d'] == 2
    assert semantics['source_only_history_3d'] is True
    assert semantics['illumination_by_dimension']['3'] == 'None only'
    examples = []
    for locale in ('zh', 'en'):
        docs = project / 'docs' / ('en' if locale == 'en' else '')
        text = (docs / 'usage.md').read_text(encoding='utf-8')
        section = text.split('(scalar-3d-example)=', 1)[1].split('(scalar-notes)=', 1)[0]
        code = re.search(r'^```python\n(.*?)^```', section, re.M | re.S)[1]
        examples.append(code)
        tree = parse_python(code)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        scalar_calls = [node for node in calls if ast.unparse(node.func) == 'starwave.scalar']
        assert len(scalar_calls) == 1, f'{locale}: expected one standalone 3D scalar call'
        values = {item.arg: item.value for item in scalar_calls[0].keywords}
        expected = {param.arg for param in parse_signature(contract['signature']).args.kwonlyargs}
        assert set(values) == expected | {'grid_spacing', 'dt'}, f'{locale}: 3D example optional arguments incomplete'
        spacing = ast.literal_eval(values['grid_spacing'])
        assert len(spacing) == 3 and min(spacing) > 0 and len(set(spacing)) > 1
        accuracy = ast.literal_eval(values['accuracy'])
        assert accuracy in (2, 4, 6, 8)
        assert ast.literal_eval(values['boundary_buffer']) >= accuracy // 2 + 1
        assert ast.literal_eval(values['pml_width']) > 0
        assert ast.literal_eval(values['memory']) == 'boundary'
        assert ast.literal_eval(values['illumination']) is None
        models = [node for node in calls if ast.unparse(node.func) == 'torch.full']
        assert len(models) == 1 and len(ast.literal_eval(models[0].args[0])) == 3
        assert any(item.arg == 'requires_grad' and ast.literal_eval(item.value) is True for item in models[0].keywords)
        assert any(isinstance(node.func, ast.Attribute) and node.func.attr == 'requires_grad_' for node in calls)
        assert any(isinstance(node.func, ast.Attribute) and node.func.attr == 'backward' for node in calls)
    assert examples[0] == examples[1], 'Bilingual 3D runnable snippets differ'


def check_gsls_contract(project, contracts):
    """Pin V16 semantics separately from historical 7.0.0 binary evidence."""
    gsls = contracts['visco_gsls']['semantics']
    assert gsls['reviewed_source_version'] == '0.1.0.dev16'
    assert gsls['publication_status'] == 'pending'
    assert gsls['public_wheel_contains_api'] is False
    assert gsls['default_mode'] == 'hao1' and gsls['hao1_mechanisms'] == 5
    assert gsls['modes'] == ['hao1', 'band_fit', 'sls_compat']
    assert gsls['memory_by_backend'] == {'cuda': ['full', 'checkpoint'], 'native_cpu': ['full', 'checkpoint'], 'torch': ['full', 'checkpoint']}
    assert gsls['density_gradient'] is False
    assert gsls['gradients'] == ['vp', 'q', 'source_amplitudes']
    assert gsls['source_sign'] == -1 and gsls['startup_weight'] == 0.5
    assert gsls['last_source_vjp'] == 0
    snippets = []
    for locale in ('zh', 'en'):
        docs = project / 'docs' / ('en' if locale == 'en' else '')
        assert not (docs / 'visco-sls.md').exists(), 'Removed standalone SLS API page persists'
        text = (docs / 'visco-gsls.md').read_text(encoding='utf-8')
        for anchor in ('function', 'physics', 'gradients', 'stability', 'example', 'runtime', 'references', 'modes', 'provider', 'memory', 'coefficients'):
            assert f'(visco-gsls-{anchor})=' in text, f'{locale}: missing GSLS section {anchor}'
        section = text.split('(visco-gsls-example)=', 1)[1].split('(visco-gsls-runtime)=', 1)[0]
        snippets.append(re.findall(r'^```python\n(.*?)^```', section, re.M | re.S))
        assert snippets[-1], f'{locale}: missing GSLS example'
        standalone = [code for code in snippets[-1] if 'import torch\nimport starwave' in code]
        assert len(standalone) == 1, f'{locale}: standalone GSLS example missing'
        example_tree = parse_python(standalone[0])
        calls = [node for node in ast.walk(example_tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'starwave.visco_gsls']
        assert len(calls) == 1, f'{locale}: standalone GSLS example call missing'
        values = {item.arg: item.value for item in calls[0].keywords}
        expected = {param.arg for param in parse_signature(contracts['visco_gsls']['signature']).args.kwonlyargs}
        assert set(values) == expected, f'{locale}: GSLS example optional arguments incomplete'
        for key, expected_value in {'mode': 'hao1', 'backend': 'torch', 'memory': 'full', 'n_mechanisms': 5, 'frequency_band': (1.0, 200.0), 'checkpoint_interval': None}.items():
            assert ast.literal_eval(values[key]) == expected_value, f'{locale}: GSLS example {key}'
    assert snippets[0] == snippets[1], 'Bilingual GSLS examples differ'


def check_torch_backend_docs(project, contracts):
    """Guard additive V16 options and full examples without importing solvers."""
    legacy = json.loads((project / 'tools' / 'api_legacy_v14_common_contract.json').read_text())
    propagators = ('scalar', 'vrz', 'vti', 'elastic', 'visco_gsls')
    added = ('backend', 'execution', 'checkpoint_interval', 'compile_steps')
    for name in propagators:
        args = parse_signature(contracts[name]['signature']).args
        options = dict((param.arg, ast.literal_eval(default)) for param, default in parameter_defaults(args) if default is not None)
        assert options['execution'] == 'eager' and options['compile_steps'] == 1
        assert options['checkpoint_interval'] is None
        assert options['backend'] == ('cuda' if name == 'visco_gsls' else None)
        assert options['memory'] == ('boundary' if name in ('scalar', 'vrz', 'vti') else 'full')
        cap = contracts[name]['torch_backend']
        assert cap == {'devices': ['cpu', 'cuda'], 'memory': ['full', 'checkpoint'],
                       'execution': ['eager', 'compile'], 'checkpoint_interval_default_internal_steps': 32,
                       'mps': False, 'amp': False, 'gradient_order': 1, 'native_default_preserved': True}
        if name != 'visco_gsls':
            original = parse_signature(legacy[name]['signature']).args
            removed = copy.deepcopy(args)
            assert tuple(param.arg for param in removed.kwonlyargs[-4:]) == added
            removed.kwonlyargs = removed.kwonlyargs[:-4]
            removed.kw_defaults = removed.kw_defaults[:-4]
            assert arguments_without_annotations(removed) == arguments_without_annotations(original), f'{name}: unreviewed old-parameter change'
    snippets = {}
    for locale in ('zh', 'en'):
        folder = project / 'docs' / ('en' if locale == 'en' else '')
        content = (folder / 'pytorch-backend.md').read_text(encoding='utf-8')
        for section in ('selection', 'memory', 'scope', 'example', 'all-options', 'performance'):
            assert f'(torch-{section})=' in content, f'{locale}: missing Torch section {section}'
        assert all(word in content for word in ('MPS', 'AMP', 'step_ratio', '32', 'compile_steps', 'boundary_buffer', 'model_gradient_sampling_interval'))
        smoke_section = content.split('(torch-example)=', 1)[1].split('(torch-all-options)=', 1)[0]
        smoke = re.search(r'^```python\n(.*?)^```', smoke_section, re.M | re.S)[1]
        snippets[locale] = [smoke]
        smoke_calls = [node for node in ast.walk(parse_python(smoke)) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'starwave.scalar']
        assert len(smoke_calls) == 1
        smoke_options = {kw.arg: kw.value for kw in smoke_calls[0].keywords}
        for key, value in {'backend': 'torch', 'memory': 'checkpoint', 'execution': 'eager', 'checkpoint_interval': 32}.items():
            assert ast.literal_eval(smoke_options[key]) == value
        for name in ('scalar', 'vrz', 'vti', 'elastic'):
            section = content.split(f'<!-- torch-api:{name}:full:start -->', 1)[1].split(f'<!-- torch-api:{name}:full:end -->', 1)[0]
            code = re.search(r'^```python\n(.*?)^```', section, re.M | re.S)[1]
            snippets[locale].append(code)
            calls = [node for node in ast.walk(parse_python(code)) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'starwave.' + name]
            assert len(calls) == 1
            args = parse_signature(contracts[name]['signature']).args
            required = [param.arg for param, default in parameter_defaults(args) if default is None and param in args.args]
            assert len(calls[0].args) == len(required)
            keywords = {kw.arg: kw.value for kw in calls[0].keywords}
            expected = {param.arg for param, _ in parameter_defaults(args)} - set(required)
            assert set(keywords) == expected, f'{locale}/{name}: incomplete all-option call'
            for key, value in {'backend': 'torch', 'execution': 'eager', 'memory': 'checkpoint', 'checkpoint_interval': 32, 'compile_steps': 1}.items():
                assert ast.literal_eval(keywords[key]) == value
    assert snippets['zh'] == snippets['en'], 'Bilingual Torch all-option calls differ'


def check(wheel_path=None, source_root=None):
    project = Path(__file__).resolve().parents[1]
    contracts = json.loads((project / 'tools' / 'api_contract.json').read_text())
    assert set(contracts) == set(MODULES), 'Public contract inventory changed'
    check_scalar3d_examples(project, contracts['scalar'])
    check_gsls_contract(project, contracts)
    check_torch_backend_docs(project, contracts)
    legacy_common = json.loads((project / 'tools' / 'api_legacy_v14_common_contract.json').read_text())
    assert set(legacy_common) == set(MODULES) - {'visco_gsls', 'prepare_visco_gsls', 'visco_gsls_native_status'}
    legacy = json.loads((project / 'tools' / 'api_legacy_v14_contract.json').read_text())['apis']
    assert set(legacy) == {'visco_sls', 'prepare_visco_sls', 'visco_sls_native_status'}
    for entry in legacy.values():
        assert set(entry) == {'signature'}, 'Legacy snapshot must remain signature-only'
        parse_signature(entry['signature'])
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
    if source_root:
        class SourceArchive:
            def read(self, path):
                return (Path(source_root) / path).read_bytes()
            def namelist(self):
                return [p.relative_to(source_root).as_posix() for p in Path(source_root).rglob('*.py')]
        source = SourceArchive()
        for name, module in MODULES.items():
            actual = wheel_function(source, module, name.rsplit('.', 1)[-1])
            assert arguments_without_annotations(languages['zh'][name]) == arguments_without_annotations(actual.args), f'{name}: differs from source'
        exports = parse_python(source.read('starwave/__init__.py').decode('utf-8'))
        public = next(ast.literal_eval(n.value) for n in exports.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__all__' for t in n.targets))
        assert {'visco_gsls', 'prepare_visco_gsls', 'visco_gsls_native_status'} <= set(public)
        assert not {'visco_sls', 'prepare_visco_sls', 'visco_sls_native_status'} & set(public), 'Removed standalone SLS APIs are exported'
        print(f'PASS: {len(MODULES)} documented signatures match the supplied source tree (not a wheel verification).')
    if wheel_path:
        wheel_path = Path(wheel_path)
        assert WHEEL_SHA256, 'Public 7.0.0 wheel identity has not yet been pinned'
        assert hashlib.sha256(wheel_path.read_bytes()).hexdigest() == WHEEL_SHA256, f'Not the verified {WHEEL_VERSION} binary wheel'
        with zipfile.ZipFile(wheel_path) as wheel:
            assert 'starwave/visco_gsls.py' not in wheel.namelist(), 'Historical wheel unexpectedly contains current GSLS API'
            metadata = BytesParser().parsebytes(wheel.read(f'starwave-{WHEEL_VERSION}.dist-info/METADATA'))
            assert metadata['Name'] == 'starwave' and metadata['Version'] == WHEEL_VERSION
            wheel_metadata = BytesParser().parsebytes(wheel.read(f'starwave-{WHEEL_VERSION}.dist-info/WHEEL'))
            assert wheel_metadata['Root-Is-Purelib'] == 'false', 'A binary wheel is required'
            assert wheel_metadata['Tag'] == 'py3-none-manylinux_2_35_x86_64'
            receipt = json.loads(wheel.read('starwave/_binary/release.json'))
            assert receipt['artifact_kind'] == 'precompiled-binary-only'
            assert receipt['contract']['package_version'] == WHEEL_VERSION
            assert receipt['contract']['source_version'] == '0.1.0.dev14'
            for kind, filename in [('core', 'libstarwave_cuda.so'), ('elastic', 'libstarwave_deepwave_elastic.so'),
                                   ('sls_cpu', 'libstarwave_sls_cpu.so'), ('sls_cuda', 'libstarwave_sls_cuda.so')]:
                library = wheel.read('starwave/_binary/' + filename)
                assert library.startswith(b'\x7fELF'), f'Expected native ELF library: {filename}'
                assert hashlib.sha256(library).hexdigest() == receipt['libraries'][kind]['sha256'], f'Binary receipt mismatch: {filename}'
            wheel_contracts = legacy_common | legacy
            wheel_modules = {name: module for name, module in MODULES.items() if 'gsls' not in name} | {name: 'starwave/visco_sls.py' for name in legacy}
            for name, module in wheel_modules.items():
                actual = wheel_function(wheel, module, name.rsplit('.', 1)[-1])
                assert arguments_without_annotations(parse_signature(wheel_contracts[name]['signature']).args) == arguments_without_annotations(actual.args), f'{name}: differs from public wheel'
                if name in WHEEL_TYPED_APIS:
                    actual_types = {param.arg: ast.unparse(param.annotation) for param, _ in parameter_defaults(actual.args)}
                    assert legacy_common[name]['types'] == actual_types, f'{name}: types differ from public wheel'
                    assert legacy_common[name]['return'] == ast.unparse(actual.returns), f'{name}: return type differs from public wheel'
    counts = {name: len(parameter_defaults(languages['zh'][name])) for name in MODULES}
    print(f'PASS: both languages: typed signatures, descriptions/defaults {counts}, {len(MODULES)} return contracts; {blocks} Python blocks parsed as Python 3.10.')
    print(f'PASS: Historical V14/common signatures match verified {WHEEL_VERSION} binary wheel; metadata and native-library receipt hashes match. V16 additions are not verified by this wheel.' if wheel_path else f'PASS: bilingual signatures match the API contract; V16 source signatures are checked only with --source-root; historical {WHEEL_VERSION} binary identity only with --wheel.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', help=f'StarWave {WHEEL_VERSION} binary wheel (read only, never imported)')
    parser.add_argument('--source-root', type=Path, help='Optional source tree for AST-only signature comparison; never imported')
    args = parser.parse_args()
    check(args.wheel, args.source_root)
