"""Check the additive qP TTI contract without importing propagation libraries."""
import argparse
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def args(signature):
    return ast.parse('def tti' + signature + ': pass', feature_version=(3, 10)).body[0].args


def check(source_root=None):
    contract = json.loads((ROOT / 'tools/tti_api_contract.json').read_text())
    expected = ast.dump(args(contract['signature']))
    examples = []
    for path in ('docs/usage.md', 'docs/en/usage.md'):
        text = (ROOT / path).read_text()
        found = re.findall(r'^```\{py:function\} starwave\.tti(\([^\n]+)\n(.*?)^```', text, re.M | re.S)
        assert len(found) == 1, f'{path}: expected one TTI directive'
        signature, body = found[0]
        assert ast.dump(args(signature)) == expected, f'{path}: signature differs'
        assert re.findall(r'^:param (\w+):', body, re.M) == contract['parameter_names'], f'{path}: parameter coverage/order'
        assert ':returns:' in body, f'{path}: missing returns'
        assert text.count('(tti)=') == 1 and '[qP VTI / TTI Function](#tti)' in text
        code = re.search(r'<!-- tti:example:start -->\s*```python\n(.*?)```\s*<!-- tti:example:end -->', text, re.S)[1]
        ast.parse(code, feature_version=(3, 10))
        examples.append(code)
    assert examples[0] == examples[1], 'Bilingual runnable examples differ'
    # Load only the stable section assignments, not build/deployment operations.
    scope = {}
    module = ast.parse((ROOT / 'tools/build_docs.py').read_text())
    for statement in module.body:
        if isinstance(statement, (ast.Assign, ast.AugAssign)):
            names = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)}
            if names & {'USAGE_SECTIONS', 'USAGE_NUMBERS', 'BORN_SECTIONS', 'V19_HELPER_SECTIONS', '_born_insert', '_vrr_ref_insert', '_elastic_vrr_insert', '_tti_insert'}:
                exec(compile(ast.Module(body=[statement], type_ignores=[]), '<section map>', 'exec'), scope)
    for path in ('docs/usage.md', 'docs/en/usage.md'):
        source_sections = ['usage'] + re.findall(r'^\(([^)]+)\)=\n#+ ', (ROOT / path).read_text(), re.M)
        assert source_sections == list(scope['USAGE_SECTIONS']), f'{path}: section order mismatch'
    numbers = dict(zip(scope['USAGE_SECTIONS'], scope['USAGE_NUMBERS']))
    assert numbers['tti'] == 74 and numbers['elastic-vrr-api'] == 72 and numbers['elastic-vrr-parameters'] == 73
    assert len(numbers) == len(scope['USAGE_NUMBERS']) == len(set(scope['USAGE_NUMBERS']))
    if source_root:
        tree = ast.parse((Path(source_root) / contract['source_path']).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tti')
        assert ast.dump(function.args) == expected, 'Current source signature differs from pinned TTI contract'
    print('PASS: bilingual TTI signature, 26 parameters, example syntax and additive anchor 74; no numerical execution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path)
    check(parser.parse_args().source_root)
