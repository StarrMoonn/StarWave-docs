# StarWave-docs · 中文 / English

StarWave 2.0.0 双语使用手册，基于 Sphinx、Read the Docs 主题和 MyST Markdown。完整中文和英文版本各含 18 个内容页面，覆盖安装、正演、反演及 API；每种语言独立构建、导航和搜索。API 采用 Deepwave 风格的类型签名、参数、返回、注意事项及示例组织，所有说明针对 StarWave 原创编写。

Bilingual user documentation for StarWave 2.0.0, built with Sphinx, the Read the Docs theme, and MyST Markdown. Each language contains the complete 18-page manual and a separate search index. The API presentation takes organizational cues from Deepwave while preserving StarWave’s own public contracts.

## 构建 / Build

Use Python 3.12. StarWave, PyTorch, CUDA and a GPU are not required for documentation builds.

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r docs/requirements.txt
python tools/build_docs.py
python tools/check_site.py _build/html
python tools/check_api_docs.py
python -m http.server 8000 --directory _build/html --bind 127.0.0.1
```

Open http://127.0.0.1:8000/zh/ or http://127.0.0.1:8000/en/. `build_docs.py` runs both Sphinx builds with warnings treated as errors. Chinese sources remain in `docs/`; English counterparts are in `docs/en/`. Shared templates, CSS and scripts are in `docs/_templates/` and `docs/_static/`. Do not use a single raw Sphinx command for the published bilingual output.

Each page has an accessible language switch. Section and Python-object anchors survive switching; the original root-level URLs, including `/api/scalar.html`, redirect to their corresponding Chinese pages while retaining query parameters and mapped legacy fragments. Without JavaScript, compatibility pages expose links to both languages. Search runs entirely in the selected language’s static index.

## 检查 / Checks

- `check_site.py`: local links, anchors, resources, complete locale coverage, independent search assets, English translation coverage, Python syntax and restricted-content guardrails.
- `check_api_docs.py`: all five typed function signatures, types/defaults/descriptions, scalar 16 / VRZ 18 / VTI 20 parameter coverage, return contracts and Python snippets. The checked-in public contract snapshot prevents cross-language agreement from hiding interface drift.
- Optional `--wheel PATH`: verifies the public 2.0.0 wheel SHA-256 and compares signatures through AST parsing, without importing or running StarWave. Wheels are never part of the source or published site.

Browser regression checks require an optional dependency and browser installation:

```bash
python -m pip install -r docs/requirements-browser.txt
python -m playwright install chromium
python tools/check_browser.py
```

The browser checks cover desktop/mobile layout, section and API switching, legacy routes, search isolation, browser history, keyboard access and no-JavaScript navigation. Screenshots are written under `_build/qa/`.

## 托管 / Hosting

`.github/workflows/docs.yml` builds and validates both languages for pushes and pull requests. Browser regression checks must pass before deployment. It retains HTML and QA screenshot artifacts. Main-branch pushes and manual workflow runs deploy to GitHub Pages after all checks; pull requests only build. GitHub Pages must already be configured to use GitHub Actions. No repository settings or credentials are changed by this project.

`.readthedocs.yaml` uses the same bilingual build as an optional alternative. Read the Docs account or hosting eligibility is not implied by this configuration.

## 内容边界 / Scope

This repository contains documentation, original synthetic usage examples and build tooling. It contains no propagation implementation, binary libraries, private datasets or internal records. It neither imports StarWave through autodoc nor publishes source pages. Public 2.0.0 API signatures are verified; target-GPU numerical behavior, performance, multi-GPU operation and full FWI convergence remain unvalidated. See the manual’s documentation-status page.

No new software or documentation license is declared on behalf of the owners.
