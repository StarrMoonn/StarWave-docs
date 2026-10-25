"""Regenerate the renamed Chinese homepage/Usage WOFF subsets.

Requires fontTools (generation only, not a documentation build dependency).
Use licensed Noto Sans CJK SC 2.004 OTFs or TTCs, for example:
    python tools/build_home_fonts.py \
        --regular /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc \
        --bold /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc
The upstream SIL OFL and copyright remain in docs/_static/fonts/OFL.txt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from fontTools import subset
from fontTools.ttLib import TTCollection, TTFont

ROOT = Path(__file__).resolve().parents[1]
FAMILY = 'StarWave Home Sans'


def source_font(path, style):
    if path.suffix.lower() in {'.ttc', '.otc'}:
        collection = TTCollection(path, lazy=False)
        matches = [f for f in collection.fonts
                   if f['name'].getDebugName(1) == 'Noto Sans CJK SC'
                   and f['name'].getDebugName(2) == style]
        if len(matches) != 1:
            raise ValueError(f'{path}: expected exactly one Noto Sans CJK SC {style} face')
        font = matches[0]
    else:
        font = TTFont(path)
    if (font['name'].getDebugName(1) != 'Noto Sans CJK SC'
            or font['name'].getDebugName(2) != style
            or not font['name'].getDebugName(5).startswith('Version 2.004;')):
        raise ValueError(f'{path}: expected licensed Noto Sans CJK SC 2.004 {style}')
    font.recalcTimestamp = False
    return font


def build(regular, bold):
    folder = ROOT / 'docs/_static/fonts'
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = re.sub(r'```\{toctree\}.*?\n```', '',
                    (ROOT / 'docs/index.md').read_text(), flags=re.S)
    source += (ROOT / 'docs/usage.md').read_text()
    characters = ''.join(sorted(set(re.findall(r'[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]', source))))
    wanted = {ord(c) for c in characters}
    assets = []
    for path, style, weight in ((regular, 'Regular', 400), (bold, 'Bold', 700)):
        font = source_font(path, style)
        if not wanted <= set(font.getBestCmap()):
            raise ValueError(f'{path}: source is missing requested characters')
        options = subset.Options()
        options.recalc_timestamp = False
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(unicodes=wanted)
        subsetter.subset(font)
        postscript = 'StarWaveHomeSans-' + style
        names = {1: FAMILY, 2: style, 3: postscript + '-Usage',
                 4: FAMILY + ' ' + style, 6: postscript}
        for record in font['name'].names:
            if record.nameID in names:
                record.string = names[record.nameID].encode(record.getEncoding())
        if 'CFF ' in font:
            cff = font['CFF '].cff
            cff.fontNames[0] = postscript
            cff.topDictIndex[0].FullName = names[4]
            cff.topDictIndex[0].FamilyName = FAMILY
        font.flavor = 'woff'
        target = folder / f'home-sans-sc-{style.lower()}.woff'
        font.save(target)
        data = target.read_bytes()
        check = TTFont(target)
        assert wanted == set(check.getBestCmap()), 'Generated cmap differs from requested subset'
        assert check['name'].getDebugName(1) == FAMILY
        assert data[:4] == b'wOFF' and len(data) <= 300000, 'Invalid/oversized generated WOFF'
        assets.append(dict(file=target.name, sha256=hashlib.sha256(data).hexdigest(),
                           size_bytes=len(data), weight=weight, characters=characters))
        print(f'{target.name}: {len(characters)} characters, {len(data)} bytes, {assets[-1]["sha256"]}')
    manifest['modification'] = ('Chinese homepage and Usage glyph subset including qP TTI and Maxwell/GPR APIs; '
                                'WOFF conversion; renamed family StarWave Home Sans; Latin and other pages keep system fonts.')
    manifest['assets'] = assets
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regular', type=Path, required=True)
    parser.add_argument('--bold', type=Path, required=True)
    args = parser.parse_args()
    build(args.regular, args.bold)
