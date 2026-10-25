"""Check bilingual Maxwell/GPR API contracts without importing propagation libraries."""
import argparse
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def signature_args(signature):
    return ast.parse('def documented' + signature + ': pass').body[0].args


def section_map():
    """Read the build's literal section assignments without running a site build."""
    scope = {}
    for statement in ast.parse((ROOT / 'tools/build_docs.py').read_text()).body:
        if isinstance(statement, (ast.Assign, ast.AugAssign)):
            names = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)}
            if names & {'USAGE_SECTIONS', 'USAGE_NUMBERS', 'BORN_SECTIONS',
                        'V19_HELPER_SECTIONS', '_born_insert', '_vrr_ref_insert',
                        '_elastic_vrr_insert', '_tti_insert', '_gpr_insert'}:
                exec(compile(ast.Module(body=[statement], type_ignores=[]),
                             '<section map>', 'exec'), scope)
    return scope['USAGE_SECTIONS'], scope['USAGE_NUMBERS']


def check(source_root=None):
    contract = json.loads((ROOT / 'tools/gpr_api_contract.json').read_text())
    expected = ast.dump(signature_args(contract['signature']))
    sections, numbers = section_map()
    assert len(sections) == len(numbers) == len(set(sections)) == len(set(numbers))
    mapping = dict(zip(sections, numbers))
    assert all(mapping[name] == number for name, number in contract['section_numbers'].items())
    assert mapping['tti'] == 74 and mapping['elastic-vrr-api'] == 72
    assert mapping['elastic-vrr-parameters'] == 73
    examples = []
    for relative in ('docs/usage.md', 'docs/en/usage.md'):
        text = (ROOT / relative).read_text()
        source_sections = ['usage'] + re.findall(r'^\(([^)]+)\)=\n#+ ', text, re.M)
        assert source_sections == list(sections), f'{relative}: section order differs'
        for name in ('gpr', 'maxwell'):
            matches = re.findall(r'^```\{py:function\} starwave\.' + name +
                                 r'(\([^\n]+)\n(.*?)^```', text, re.M | re.S)
            assert len(matches) == 1, f'{relative}: missing/duplicate {name} directive'
            signature, body = matches[0]
            assert ast.dump(signature_args(signature)) == expected, f'{relative}: {name} signature drift'
            assert ':returns:' in body and ':rtype:' in body
            if name == 'gpr':
                assert re.findall(r'^:param (\w+):', body, re.M) == contract['parameter_names']
                assert re.findall(r'^:type (\w+):', body, re.M) == contract['parameter_names']
        for token in ('[Maxwell / GPR Function](#gpr)', contract['source_feature_commit'],
                      'min(s,nt-t)', 'callback_requires_grad=True', 'magnetic_time=(step-0.5)*dt',
                      'python compile_all.py --backend cpu --equation maxwell',
                      'python compile_all.py --backend cuda --equation maxwell --arch 80,89',
                      '--equation gpr', '--equation all'):
            assert token in text, f'{relative}: missing {token}'
        assert re.search(r'\(gpr\)=.*\(gpr-scope\)=.*\(native-runtime\)=', text, re.S)
        code = re.search(r'<!-- gpr:example:start -->\s*```python\n(.*?)```\s*<!-- gpr:example:end -->', text, re.S)[1]
        ast.parse(code)
        assert code.startswith('import starwave\n'), f'{relative}: use installed import first'
        assert all(bad not in code for bad in ('sys.path', 'prepare_native', 'compile_all', 'torch.compile('))
        examples.append(code)
    assert examples[0] == examples[1], 'Bilingual runnable examples differ'
    if source_root:
        for name, path in contract['source_paths'].items():
            tree = ast.parse((source_root / path).read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            assert ast.dump(function.args) == expected, f'{name}: actual source signature differs'
    print(f"PASS: bilingual Maxwell/GPR signatures, {len(contract['parameter_names'])} parameters, example syntax, stable anchors 75–82 and source signatures when requested; no numerical execution.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    check(parser.parse_args().source_root)
