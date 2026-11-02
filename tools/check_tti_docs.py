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
    from build_docs import usage_section_map
    sections, section_numbers = usage_section_map()
    for locale in ('zh', 'en'):
        assert usage_section_map(locale) == (sections, section_numbers), 'Translated Usage anchors/order differ'
    numbers = dict(zip(sections, section_numbers))
    assert numbers['tti'] == 74 and numbers['elastic-vrr-api'] == 72 and numbers['elastic-vrr-parameters'] == 73
    assert len(numbers) == len(section_numbers) == len(set(section_numbers))
    if source_root:
        tree = ast.parse((Path(source_root) / contract['source_path']).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tti')
        assert ast.dump(function.args) == expected, 'Current source signature differs from pinned TTI contract'
    print('PASS: bilingual TTI signature, 26 parameters, example syntax and additive anchor 74; no numerical execution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path)
    check(parser.parse_args().source_root)
