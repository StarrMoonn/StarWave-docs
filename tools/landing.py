"""Bounded, hash-verified public landing assets; never copy the private Site checkout."""
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'landing'

def approved_assets():
    entries = json.loads((SOURCE / 'manifest.json').read_text())['sha256']
    actual = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != set(entries):
        raise ValueError('Landing manifest does not exactly match source files')
    result = set()
    for name, expected in entries.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError(f'Invalid landing asset path: {name}')
        path = SOURCE / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Landing asset hash mismatch: {name}')
        result.add(path)
    return result

def publish(destination, home_fragments):
    for source in approved_assets():
        target = destination / source.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Preserve old root home bookmarks while leaving all new landing anchors alone.
    script = ('const legacyHomeFragments=' + json.dumps(home_fragments, ensure_ascii=True) + ';\n'
              'let oldHomeFragment=location.hash.slice(1);\n'
              'try{oldHomeFragment=decodeURIComponent(oldHomeFragment);}catch(_){}\n'
              'if(Object.prototype.hasOwnProperty.call(legacyHomeFragments,oldHomeFragment)){' 
              'const target=new URL("zh/index.html",location.href);target.search=location.search;'
              'target.hash=legacyHomeFragments[oldHomeFragment];location.replace(target.href);}\n')
    (destination / 'legacy-home.js').write_text(script)
    index = destination / 'index.html'
    index.write_text(index.read_text().replace('</head>', '<script src="legacy-home.js"></script></head>'))

def check(destination):
    errors = []
    for source in approved_assets():
        target = destination / source.relative_to(SOURCE)
        if source.name == 'index.html':
            expected = source.read_text().replace('</head>', '<script src="legacy-home.js"></script></head>')
            if not target.is_file() or target.read_text() != expected:
                errors.append('Landing index does not match approved source')
        elif not target.is_file() or target.read_bytes() != source.read_bytes():
            errors.append(f'Landing asset mismatch: {source.name}')
    if not (destination / 'legacy-home.js').is_file():
        errors.append('Missing legacy home bookmark route')
    return errors
