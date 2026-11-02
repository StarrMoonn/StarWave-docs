"""Publish only the reviewed static Agent introduction, never private implementation."""
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parents[1] / 'agent'

def approved_assets():
    entries = json.loads((SOURCE / 'manifest.json').read_text())['sha256']
    actual = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != set(entries):
        raise ValueError('Agent introduction manifest must exactly match public assets')
    result = set()
    for name, expected in entries.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid Agent introduction path: {name}')
        path = SOURCE / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Agent introduction hash mismatch: {name}')
        result.add(path)
    return result

def publish(destination):
    for source in approved_assets():
        target = destination / 'agent' / source.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

def check(destination):
    errors = []
    for source in approved_assets():
        target = destination / 'agent' / source.relative_to(SOURCE)
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            errors.append(f'Agent introduction mismatch: {source.name}')
    return errors
