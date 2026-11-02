"""Check current VTI-VRR separately from immutable historical API snapshots."""
import argparse
import ast
import json
from pathlib import Path
import re

from check_gpr_docs import section_map

ROOT = Path(__file__).resolve().parents[1]


def arguments(signature):
    return ast.parse('def documented' + signature + ': pass').body[0].args


def check(source_root=None):
    contract = json.loads((ROOT / 'tools/vti_vrr_api_contract.json').read_text())
    expected = ast.dump(arguments(contract['signature']))
    sections, numbers = section_map()
    assert len(sections) == len(numbers) == len(set(sections)) == len(set(numbers))
    mapping = dict(zip(sections, numbers))
    assert all(mapping[key] == value for key, value in contract['section_numbers'].items())
    examples = []
    for relative in ('docs/usage.md', 'docs/en/usage.md'):
        text = (ROOT / relative).read_text()
        matches = re.findall(r'^```\{py:function\} starwave\.vti_vrr(\([^\n]+)\n(.*?)^```', text, re.M | re.S)
        assert len(matches) == 1, f'{relative}: missing/duplicate VTI-VRR directive'
        signature, body = matches[0]
        assert ast.dump(arguments(signature)) == expected, f'{relative}: VTI-VRR signature/default drift'
        assert re.findall(r'^:param (\w+):', body, re.M) == contract['parameter_names']
        function = ast.parse('def documented' + signature + ': pass').body[0]
        assert ast.unparse(function.returns) == 'tuple[torch.Tensor, ...]'
        assert ':returns:' in body
        assert ['usage'] + re.findall(r'^\(([^)]+)\)=\n#+ ', text, re.M) == list(sections)
        section = text.split('(vti-vrr)=', 1)[1].split('(tti)=', 1)[0]
        for token in ('receiver_fields', 'density=Z/c', '(n+1)*user_dt', 'float32/float64',
                      'D_i(c)/c-2*R_i', 'compile_steps=2', 'checkpoint_interval=32'):
            assert token in section, f'{relative}: missing contract {token}'
        code = re.findall(r'^```python\n(.*?)^```', section, re.M | re.S)
        assert len(code) == 1, f'{relative}: missing/duplicate runnable example'
        tree = ast.parse(code[0])
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == 'starwave.vti_vrr']
        assert len(calls) == 1
        options = {kw.arg: kw.value for kw in calls[0].keywords}
        assert ast.literal_eval(options['receiver_fields']) == ('sH', 'sV', 'vx', 'vz')
        assert ast.literal_eval(options['backend']) == 'torch'
        assert 'density' in options and 'rx' in options and 'rz' in options
        examples.append(code[0])
    assert examples[0] == examples[1], 'Bilingual examples differ'
    if source_root:
        tree = ast.parse((Path(source_root) / contract['source_path']).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'vti_vrr')
        assert ast.dump(function.args) == expected, 'Source VTI-VRR signature differs'
    print('PASS: VTI-VRR bilingual signature/defaults, parameter order, tuple return, '
          'anchors 83–86, identical example syntax and optional source AST.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    check(parser.parse_args().source_root)
