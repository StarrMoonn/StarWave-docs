"""Validate the independent source-extension manual without importing StarWave."""
import argparse
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def signature_args(signature):
    return ast.parse('def documented' + signature + ': pass', feature_version=(3, 10)).body[0].args


def check(source_root=None):
    contract = json.loads((ROOT / 'tools/vrr_api_contract.json').read_text())
    expected = signature_args(contract['signature'])
    names = [arg.arg for arg in expected.args + expected.kwonlyargs]
    defaults = ['required'] * len(expected.args) + [
        'required' if value is None else ast.unparse(value) for value in expected.kw_defaults]
    examples = []
    for locale, folder in [('zh', ROOT / 'docs'), ('en', ROOT / 'docs/en')]:
        usage = (folder / 'usage.md').read_text()
        text = usage.split('(vrr)=\n## VRR Function', 1)[1].split('(vti)=', 1)[0]
        blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
        assert len(blocks) == 1, f'{locale}: executable example required'
        for block in blocks:
            ast.parse(block, feature_version=(3, 10))
        signature, body = re.search(
            r'^```\{py:function\} starwave_vrr\.vrr([^\n]+)\n(.*?)^```',
            text, re.M | re.S).groups()
        documented = ast.parse('def f' + signature + ': pass', feature_version=(3, 10)).body[0]
        pinned = ast.parse('def f' + contract['documented_signature'] + ': pass', feature_version=(3, 10)).body[0]
        assert ast.dump(documented) == ast.dump(pinned), f'{locale}: typed signature drift'
        for arg in documented.args.args + documented.args.kwonlyargs:
            arg.annotation = None
        assert ast.dump(documented.args) == ast.dump(expected), f'{locale}: signature/default drift'
        fields = re.findall(r'^:param (\w+): (.+)$', body, re.M)
        assert [name for name, _ in fields] == names, f'{locale}: missing/reordered parameter descriptions'
        parameter_types = dict(re.findall(r'^:type (\w+): `(.+)`$', body, re.M))
        assert parameter_types == contract['types'], f'{locale}: parameter type drift'
        for (name, description), wanted in zip(fields, defaults):
            if wanted == 'required':
                assert ('Required' if locale == 'en' else '必填') in description
            else:
                value = re.search(r'`([^`]+)`', description).group(1)
                assert ast.literal_eval(value) == ast.literal_eval(wanted), f'{locale}: {name} default drift'
        assert ':returns:' in body and ':rtype: tuple[torch.Tensor]' in body
        example = ast.parse(blocks[0], feature_version=(3, 10))
        calls = [node for node in ast.walk(example) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'vrr']
        assert len(calls) == 1 and len(calls[0].args) == 3
        assert [kw.arg for kw in calls[0].keywords] == [arg.arg for arg in expected.kwonlyargs]
        values = {kw.arg: kw.value for kw in calls[0].keywords}
        for key, value in {'backend': 'torch', 'memory': 'checkpoint', 'execution': 'eager',
                           'checkpoint_interval': 10, 'compile_steps': 1}.items():
            assert ast.literal_eval(values[key]) == value
        examples.append(ast.dump(example))
        for required in ('starwave-vrr', 'starwave_vrr', '7.1.0', 'extensions/vrr',
                         'ABI 3', '[w,w,0,w]', 'MPS', 'float32/float64',
                         contract['source_dev_commit'], contract['source_feature_commit']):
            assert required in text, f'{locale}: missing scope marker {required}'
        assert 'VRR Function <vrr>' not in (folder / 'index.md').read_text()
        assert '- [VRZ Function](#vrz)\n- [VRR Function](#vrr)\n- [VTI Function](#vti)' in usage
        assert usage.index('(vrz)=') < usage.index('(vrr)=') < usage.index('(vti)=')
        assert not (folder / 'vrr.md').exists(), 'Standalone VRR source page must be retired'
    assert examples[0] == examples[1], 'Bilingual example drift'
    if source_root:
        source = Path(source_root) / contract['source_file']
        module = ast.parse(source.read_text(), feature_version=(3, 10))
        actual = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'vrr')
        assert ast.dump(actual.args) == ast.dump(expected), 'Authorized source signature drift'
        print('PASS: independent VRR signature matches supplied source via AST; no runtime imported.')
    print(f'PASS: independent VRR bilingual signature, {len(names)} parameters/defaults, example and source-only scope.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    check(parser.parse_args().source_root)
