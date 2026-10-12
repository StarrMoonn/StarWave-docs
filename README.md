# StarWave-docs · 中文 / English

StarWave V19 源码契约双语使用手册，使用 Sphinx、Read the Docs 主题与 MyST Markdown；每种语言独立构建、导航和搜索。

Bilingual documentation for the StarWave V19 source contract. The complete Chinese and English manuals have independent builds, navigation and search.

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) 已发布；本手册对应冻结源码 `0.1.0.dev19`、提交 `46d32877e3336d531520a696d5f2e074e6f4ef74`。[PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) 已发布，对应 V19 源码契约并包含集成 VRR。现有 [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) 仍对应 V18，不含集成 VRR。

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) is published. This manual follows frozen source `0.1.0.dev19`, commit `46d32877e3336d531520a696d5f2e074e6f4ef74`. [PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) is published with the V19 source contract and integrated VRR. The existing [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) remains V18 and excludes integrated VRR.

## V19 integrated package and examples

The current main package exposes scalar, Born, VRZ, VRR, VTI, elastic and GSLS propagation. Root installation and `compile_all.py` cover all five native groups including VRR; use `starwave.vrr`, not a separate extension installation. V19 preserves the 23-parameter Scalar API and its differentiable terminal output; callbacks remain detached diagnostics.

The authorized source contains paired `notebook/Scalar/Scalar2D_TargetWavefield_CUDA.py` / `.ipynb` and `notebook/Scalar/Scalar2D_TargetWavefield_TorchCompile.py` / `.ipynb`. Historical tutorial assets remain unchanged. PyPI 9.1.0 is published with verified artifact identity and CPU checks; the historical 9.0.0 wheel remains V18 without VRR.

## Release channels and evidence

Use the current [Chinese installation guide](docs/installation.md) or [English installation guide](docs/en/installation.md). Historical V18 and earlier release receipts, binary hashes, migration instructions and validation limits are preserved in the bilingual [Release Notes](docs/en/release-notes.md). Source publication, binary publication, CPU checks and GPU numerical acceptance are separate evidence.

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
python tools/check_v19_api_contracts.py --check tools/v19_api_contract.json --docs docs
python tools/check_scalar_dev_docs.py
python tools/check_born_docs.py
python tools/check_vrr_docs.py
python tools/check_tutorial_assets.py
python -m http.server 8000 --directory _build/html --bind 127.0.0.1
```

Open http://127.0.0.1:8000/zh/ or http://127.0.0.1:8000/en/. `build_docs.py` runs both Sphinx builds with warnings treated as errors. Chinese sources remain in `docs/`; English counterparts are in `docs/en/`. Shared templates, CSS and scripts are in `docs/_templates/` and `docs/_static/`. Do not use a single raw Sphinx command for the published bilingual output.

The homepage introduces differentiable wave propagation with the supplied StarWave logo, three concise capabilities, a PyTorch/autograd fragment, and direct installation, Quickstart and Usage links. The large homepage headline is sans-serif; English reading text and smaller headings use a Georgia system stack with a serif fallback, following the Deepwave reading hierarchy at roughly 17px. Code stays monospace. The homepage uses original wave, gradient and network icons instead of a comparison table or numbered feature labels. Small locally served Noto Sans CJK SC glyph subsets keep the Chinese homepage and Usage sans-serif across systems; their full OFL 1.1 license and provenance are included in `docs/_static/fonts/`. English article text follows the requested serif hierarchy; the sidebar retains the RTD theme typography. The selected logo SVGs and favicon are unchanged user-provided assets, checked for safe SVG content and hashes; no broader logo license is declared.

The left sidebar uses short, single-line chapter labels and includes an About StarWave overview with SWEEP/Deepwave attribution. It is a flat list of major chapters, with a prominent Search docs field and a left-side clickable submit icon and no expand/collapse buttons. Usage links the scalar, Born, VRZ, integrated VRR, VTI, elastic, GSLS, material-conversion and native-helper documentation on its main page. Forward Modeling includes an Acquisition guide with an original two-shot geometry diagram and a combined Wave propagation chapter whose VRZ/VTI text comes from single-source includes. Docker has an explicitly unreleased installation placeholder; no container image or command is advertised. Other chapters have local page directories.

Each page has an accessible language switch. Section and Python-object anchors survive switching; the original root-level URLs, including `/api/scalar.html`, redirect to their corresponding Chinese pages (old API pages now map directly to Usage) while retaining query parameters and mapped legacy fragments. Original VRZ/VTI modeling URLs also redirect to their sections within Wave propagation, without duplicate search entries. Without JavaScript, compatibility pages expose links to both languages. Search runs entirely in the selected language’s static index.

## 检查 / Checks

- `check_v19_api_contracts.py`: 31-callable frozen-source inventory, current bilingual signature/name/kind/default checks and parameter coverage; optional source path performs AST comparison without importing StarWave.
- `check_site.py`: local links, anchors, resources, complete locale coverage, independent search assets, English translation coverage, Python syntax and restricted-content guardrails.
- `check_presentation_asset.py` (also run by `check_site.py`): the reviewed PDF's SHA-256, exactly 30 slides, an image hash for each reviewed watermark-bearing page, absence of active PDF actions or embedded files, and a strict publication allowlist. These checks preserve the independently reviewed pixels; they do not claim to recognize a watermark automatically.
- `check_scalar_dev_docs.py`: isolated current V19 scalar contract, exactly three additive options, preserved 20-parameter release snapshot, source AST and bilingual terminal-only checkpoint example.
- `check_born_docs.py`: preserved Born signature snapshot with V19 source AST and file-hash verification for scalar_born / ScalarBorn / forward (47/13/41 parameters), exact defaults/types/returns, bilingual checkpoint example and V19 scope; optional `--source-root` checks the reviewed source file SHA-256 and AST without importing it. The 12 original API snapshots remain unchanged.
- `check_vrr_docs.py`: integrated VRR signature/defaults, parameter coverage, bilingual examples and V19 scope; optional `--source-root` compares the authorized local source through AST without importing it.
- `check_api_docs.py`: typed function signatures, types/defaults/descriptions, historical scalar 20 / current scalar 23 / VRZ 22 / VTI 24 / elastic 66 parameter coverage, material conversions, native helpers, GSLS contracts, return contracts and Python snippets. The checked-in public contract snapshot prevents cross-language agreement from hiding interface drift.
- `check_tutorial_assets.py`: clean notebook/script parity, numerical-source provenance, safe metadata, configuration and measured-result invariants.
- Optional `--wheel PATH`: verifies the pinned 7.1.0 binary wheel SHA-256, package/platform metadata, native-library receipt hashes, and signatures through AST parsing, without importing or running StarWave. Wheels are never part of the source or published site.

Usage uses a 40.8px regular Georgia page title and single 30.6px Scalar Function, Born Function, VRZ Function, VRR Function, VTI Function, Elastic Function, and GSLS Function headings in both languages. Parameter names, types, defaults, and prose all use 17px text. Signatures retain 17px naturally wrapping monospace text and bold 18.7px function names; the propagator overview uses matching 17px API links, dimension tokens, and notation with consistent baselines. Code blocks and other inline code outside API fields retain their separate 15.3px size. Original section bookmarks, positional aliases, language switching, and legacy API routes are preserved.

Browser regression checks include the homepage logo, desktop/mobile layout, typography, text contrast, keyboard targets, calls to action and screenshots in both languages. They also retain the full documentation regression suite.

Browser regression checks require an optional dependency and browser installation:

```bash
python -m pip install -r docs/requirements-browser.txt
python -m playwright install chromium
python tools/check_browser.py
```

The browser checks cover desktop/mobile layout, section and API switching, legacy routes, search isolation, browser history, keyboard access and no-JavaScript navigation. Presentation checks cover the same-origin PDF response and hash, embedded-viewer configuration, real new-window and download links, and mobile/no-JavaScript alternatives in both languages. Actual inline PDF rendering depends on the browser's built-in reader. Screenshots are written under `_build/qa/`.

## 托管 / Hosting

`.github/workflows/docs.yml` builds and validates both languages for pushes and pull requests. Browser regression checks must pass before deployment. It retains HTML and QA screenshot artifacts. Main-branch pushes and manual workflow runs deploy to GitHub Pages after all checks; pull requests only build. GitHub Pages must already be configured to use GitHub Actions. No repository settings or credentials are changed by this project.

`.readthedocs.yaml` uses the same bilingual build as an optional alternative. Read the Docs account or hosting eligibility is not implied by this configuration.

## 内容边界 / Scope

This repository contains documentation, original synthetic usage examples and build tooling. It contains no propagation implementation, binary libraries, private datasets or internal records. It neither imports StarWave through autodoc nor publishes source pages. Main-package API documentation follows the V19 source contract, including integrated VRR, Born, the 23-parameter scalar API, equation-specific boundaries and import-time initialization; the V16 / 7.1.0 wheel contract remains a separately identified baseline; historical release records retain their distinct evidence scope; the three original tutorials now include measured A30 / StarWave 0.1.0.dev9 results and clean notebook downloads. Those historical tutorials do not establish public-wheel GPU validation, full adjoint correctness or general performance and convergence. The new Scalar3D Example chapter separately reports bounded source-build numerical checks and measured four-GPU experiments, including incomplete surface-only recovery. See the manual’s documentation-status page.

No new software or documentation license is declared on behalf of the owners.

The Presentation chapter publishes only the approved original 30-slide watermarked PDF. The editable slide deck, authoring files, and reference presentations are not distributed in this repository or the built site. The PDF has no new open-license grant. Visible watermarks help identify the source; they are not an unremovable protection mechanism. No external PDF viewer service or JavaScript viewer bundle is used.

## Historical tutorials and examples

The original 0.1.0.dev9 tutorials, the bounded Scalar3D experiments and the reconstruction chapter retain their historical evidence and asset hashes. They are not rerun or relabelled as V19 results. See [Status](docs/en/status.md), [Scalar3D examples](docs/en/examples/index.md), and [Release Notes](docs/en/release-notes.md) for scope and history.
