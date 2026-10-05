"""Check a built documentation site and the deliberately small source package."""
import ast
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "name" in attrs:
            self.ids.add(attrs["name"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])


def check(root):
    project = Path(__file__).resolve().parents[1]
    root = root.resolve()
    errors = []
    pages = {p: Page(p.read_text(encoding="utf-8")) for p in root.rglob("*.html")}
    if not pages or not (root / "index.html").is_file():
        errors.append("Missing HTML output or index.html")
    if not (root / ".nojekyll").is_file():
        errors.append("Missing .nojekyll for portable Pages output")
    # Each locale must be a complete site with its own index and no source dump.
    localized = {}
    for locale in ('zh', 'en'):
        folder = root / locale
        localized[locale] = {p.relative_to(folder) for p in folder.rglob('*.html')}
        for required in ('index.html', 'search.html', 'genindex.html', 'searchindex.js'):
            if not (folder / required).is_file():
                errors.append(f'Missing {locale} output: {required}')
        if (folder / '_sources').exists() and any((folder / '_sources').rglob('*')):
            errors.append(f'Unexpected {locale} published source directory')
        for relative in localized[locale]:
            path = folder / relative
            content = path.read_text(encoding='utf-8')
            language = 'zh-CN' if locale == 'zh' else 'en'
            if f'<html lang="{language}"' not in content:
                errors.append(f'Wrong document language: {locale}/{relative}')
            if 'data-sw-language="zh"' not in content or 'data-sw-language="en"' not in content:
                errors.append(f'Missing language switch: {locale}/{relative}')
            if locale == 'en' and re.search(r'[\u4e00-\u9fff]', re.sub(r'<a [^>]*data-sw-language="zh"[^>]*>中文</a>', '', content)):
                errors.append(f'Untranslated Chinese in English HTML: {relative}')
    if localized['zh'] != localized['en']:
        errors.append('Localized HTML page coverage differs')
    for relative in localized['zh'] & localized['en']:
        zh = pages[root / 'zh' / relative]
        en = pages[root / 'en' / relative]
        if {x for x in zh.ids if x.startswith('sw-section-')} != {x for x in en.ids if x.startswith('sw-section-')}:
            errors.append(f'Localized section anchors differ: {relative}')
        if {x for x in zh.ids if x.startswith('starwave.')} != {x for x in en.ids if x.startswith('starwave.')}:
            errors.append(f'Localized API anchors differ: {relative}')
        redirect = root / relative
        if not redirect.is_file() or 'location.replace' not in redirect.read_text(encoding='utf-8'):
            errors.append(f'Missing compatibility route: {relative}')
    contracts = json.loads((project / 'tools' / 'api_contract.json').read_text())
    for locale in ('zh', 'en'):
        for name in contracts:
            relative = f'api/{name}.html' if name in {'scalar', 'vrz', 'vti'} else 'api/index.html'
            if f'starwave.{name}' not in pages[root / locale / relative].ids:
                errors.append(f'Missing public API anchor: {locale}/{name}')
    links = 0
    for path, page in pages.items():
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            links += 1
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(root):
                errors.append(f"Escaping link: {path.name}: {href}")
            elif not target.exists():
                errors.append(f"Missing target: {path.name}: {href}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f"Missing anchor: {path.name}: {href}")
    # These checks are a guardrail, not a substitute for human content review.
    forbidden = [
        re.compile(r"[A-Za-z]:[\\/](?:Users|home)[\\/]", re.I),
        re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
        re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
        re.compile(r"(?:RELEASE_VERIFICATION|MAINTAINER_HANDOFF|HESS_JOINT_VP_WATER_FIX)"),
    ]
    allowed = {".md", ".py", ".txt", ".css", ".yml", ".yaml", ".html", ".js", ".json"}
    sources = []
    for p in project.rglob("*"):
        relative = p.relative_to(project)
        if any(part in {".git", ".venv", "_build", "_readthedocs", "__pycache__"} for part in relative.parts):
            continue
        if not p.is_file():
            continue
        sources.append(p)
        if p.name != ".gitignore" and p.suffix not in allowed:
            errors.append(f"Unexpected source file: {relative}")
        if p.suffix == ".py":
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                errors.append(f"Python syntax: {relative}: {exc.msg}")
        if p != Path(__file__).resolve():
            text = p.read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in forbidden):
                errors.append(f"Review restricted content: {relative}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in {".html", ".js", ".txt", ".py", ".css"}:
            if any(pattern.search(p.read_text(encoding="utf-8")) for pattern in forbidden):
                errors.append(f"Review generated content: {p.relative_to(root)}")
    if (root / "_sources").exists() and any((root / "_sources").rglob("*")):
        errors.append("Unexpected published source directory")
    if (root / ".doctrees").exists():
        errors.append("Build cache inside publication directory; use -d _build/doctrees")
    if errors:
        print("\n".join(sorted(set(errors))))
        return 1
    print(f"PASS: {len(pages)} HTML pages, {links} local links/resources, {len(sources)} source files; syntax/content checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check(Path(sys.argv[1] if len(sys.argv) > 1 else "_build/html")))
