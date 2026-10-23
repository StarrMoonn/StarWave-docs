"""Check bilingual elastic VRR signatures and parameter prose, without solver imports."""
import argparse
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def check(source=None):
    contract = json.loads((ROOT / 'tools/elastic_vrr_api_contract.json').read_text())
    for relative in ('docs/usage.md', 'docs/en/usage.md'):
        text = (ROOT / relative).read_text()
        for name, signature in contract['signatures'].items():
            expected = ast.parse(f'def f({signature}): pass').body[0].args
            pattern = r'^```\{py:function\} starwave\.' + name + r'\(([^\n]+)\)\n(.*?)^```'
            matches = re.findall(pattern, text, re.M | re.S)
            assert len(matches) == 1, (relative, name, 'missing or duplicate')
            actual = ast.parse(f'def f({matches[0][0]}): pass').body[0].args
            assert ast.dump(actual) == ast.dump(expected), (relative, name, 'signature drift')
            names = [a.arg for a in expected.args + expected.kwonlyargs]
            assert re.findall(r'^:param (\w+):', matches[0][1], re.M) == names
            assert re.findall(r'^:type (\w+):', matches[0][1], re.M) == names
            assert ':returns:' in matches[0][1] and ':rtype:' in matches[0][1]
        for marker in ('(elastic-vrr-api)=', '(elastic-vrr-parameters)=', '(#elastic-vrr-api)',
                       'examples/elastic-vrr.md', contract['source_commit']):
            assert marker in text, (relative, marker)
    if source:
        nodes = {n.name: n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)}
        for name, signature in contract['signatures'].items():
            expected = ast.parse(f'def f({signature}): pass').body[0].args
            assert ast.dump(nodes[name].args) == ast.dump(expected), name
    print('PASS: bilingual elastic VRR API signatures, defaults, parameter order, returns and links.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    check(parser.parse_args().source)
