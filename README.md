# StarWave-docs · 中文 / English

StarWave 6.0.0 双语使用手册，基于 Sphinx、Read the Docs 主题和 MyST Markdown。完整中文和英文版本各含 25 个内容页面，覆盖安装、正演、反演、API 与演示文稿；每种语言独立构建、导航和搜索。API 采用 Deepwave 风格的类型签名、参数、返回、注意事项及示例组织，所有说明针对 StarWave 原创编写。

Bilingual user documentation for StarWave 6.0.0, built with Sphinx, the Read the Docs theme, and MyST Markdown. Each language contains the complete 25-page manual and a separate search index. The API presentation takes organizational cues from Deepwave while preserving StarWave’s own public contracts.

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
python tools/check_tutorial_assets.py
python -m http.server 8000 --directory _build/html --bind 127.0.0.1
```

Open http://127.0.0.1:8000/zh/ or http://127.0.0.1:8000/en/. `build_docs.py` runs both Sphinx builds with warnings treated as errors. Chinese sources remain in `docs/`; English counterparts are in `docs/en/`. Shared templates, CSS and scripts are in `docs/_templates/` and `docs/_static/`. Do not use a single raw Sphinx command for the published bilingual output.

The homepage introduces differentiable wave propagation with the supplied StarWave logo, three concise capabilities, a PyTorch/autograd fragment, and direct installation, Quickstart and Usage links. The large homepage headline is sans-serif; English reading text and smaller headings use a Georgia system stack with a serif fallback, following the Deepwave reading hierarchy at roughly 17px. Code stays monospace. The homepage uses original wave, gradient and network icons instead of a comparison table or numbered feature labels. Small locally served Noto Sans CJK SC glyph subsets keep the Chinese homepage and Usage sans-serif across systems; their full OFL 1.1 license and provenance are included in `docs/_static/fonts/`. English article text follows the requested serif hierarchy; the sidebar retains the RTD theme typography. The selected logo SVGs and favicon are unchanged user-provided assets, checked for safe SVG content and hashes; no broader logo license is declared.

The left sidebar uses short, single-line chapter labels and includes an About StarWave overview with SWEEP/Deepwave attribution. It is a flat list of major chapters, with a prominent Search docs field and a left-side clickable submit icon and no expand/collapse buttons. Usage is a single page: its bordered directory jumps to the full scalar, VRZ, VTI, elastic, material-conversion, and native-helper documentation below. Forward Modeling includes an Acquisition guide with an original two-shot geometry diagram and a combined Wave propagation chapter whose VRZ/VTI text comes from single-source includes. Docker has an explicitly unreleased installation placeholder; no container image or command is advertised. Other chapters have local page directories.

Each page has an accessible language switch. Section and Python-object anchors survive switching; the original root-level URLs, including `/api/scalar.html`, redirect to their corresponding Chinese pages (old API pages now map directly to Usage) while retaining query parameters and mapped legacy fragments. Original VRZ/VTI modeling URLs also redirect to their sections within Wave propagation, without duplicate search entries. Without JavaScript, compatibility pages expose links to both languages. Search runs entirely in the selected language’s static index.

## 检查 / Checks

- `check_site.py`: local links, anchors, resources, complete locale coverage, independent search assets, English translation coverage, Python syntax and restricted-content guardrails.
- `check_presentation_asset.py` (also run by `check_site.py`): the reviewed PDF's SHA-256, exactly 30 slides, an image hash for each reviewed watermark-bearing page, absence of active PDF actions or embedded files, and a strict publication allowlist. These checks preserve the independently reviewed pixels; they do not claim to recognize a watermark automatically.
- `check_api_docs.py`: all nine typed function signatures, types/defaults/descriptions, scalar 16 / VRZ 18 / VTI 20 / elastic 62 parameter coverage, two four-parameter material conversions and prepare_elastic, return contracts and Python snippets. The checked-in public contract snapshot prevents cross-language agreement from hiding interface drift.
- `check_tutorial_assets.py`: clean notebook/script parity, numerical-source provenance, safe metadata, configuration and measured-result invariants.
- Optional `--wheel PATH`: verifies the pinned 6.0.0 binary wheel SHA-256, package/platform metadata, native-library receipt hashes, and signatures through AST parsing, without importing or running StarWave. Wheels are never part of the source or published site.

Usage uses a 40.8px regular Georgia page title and single 30.6px Scalar Function, VRZ Function, VTI Function, and Elastic Function headings in both languages. Parameter names, types, defaults, and prose all use 17px text. Signatures retain 17px naturally wrapping monospace text and bold 18.7px function names; the propagator overview uses matching 17px API links, dimension tokens, and notation with consistent baselines. Code blocks and other inline code outside API fields retain their separate 15.3px size. Original section bookmarks, positional aliases, language switching, and legacy API routes are preserved.

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

This repository contains documentation, original synthetic usage examples and build tooling. It contains no propagation implementation, binary libraries, private datasets or internal records. It neither imports StarWave through autodoc nor publishes source pages. Public API signatures are verified against the pinned 6.0.0 binary wheel and retain the 5.0.0 contract; the three original tutorials now include measured A30 / StarWave 0.1.0.dev9 results and clean notebook downloads. Those historical tutorials do not establish public-wheel GPU validation, full adjoint correctness or general performance and convergence. The new Scalar3D Example chapter separately reports bounded source-build numerical checks and measured four-GPU experiments, including incomplete surface-only recovery. See the manual’s documentation-status page.

No new software or documentation license is declared on behalf of the owners.

The Presentation chapter publishes only the approved original 30-slide watermarked PDF. The editable slide deck, authoring files, and reference presentations are not distributed in this repository or the built site. The PDF has no new open-license grant. Visible watermarks help identify the source; they are not an unremovable protection mechanism. No external PDF viewer service or JavaScript viewer bundle is used.

## Elastic documentation maintenance · 2026-10-08

The existing single-page Usage now documents the published V11 / 4.0.0 elastic contract, explicit material conversions, and the separate elastic preparation helper. Source/wheel signature parity and exact return order were reviewed without executing GPU propagation. Existing scalar/VRZ/VTI API content, historical tutorial assets, styling, attribution and compatibility anchors are preserved. Review scope and the published wheel hash are recorded in each language's Status and Release Notes pages. For the current binary archive, use `python tools/check_api_docs.py --wheel PATH_TO_6.0.0_WHEEL` for optional local verification; do not commit the wheel.

## Wavefield Reconstruction chapter

The bilingual Forward Modeling navigation now includes Reconstruction / 波场反传重建. The chapter derives the pinned 4.0.0 elastic boundary method, distinguishes forward reconstruction from the discrete adjoint, documents the physical velocity/traction tape and temporal-filter transpose, and gives exact payload formulas with sampling and validation limits. Four original vector schematics per language are reproducible with `tools/draw_reconstruction.py`; outlined text keeps labels consistent without a network font dependency. Keyboard-scrollable figure regions preserve readable labels on narrow screens. No software source or measured GPU results were changed.

## Scalar3D API update · 2026-10-08

The 5.0.0 manual adds source-verified 2D/3D scalar dispatch, model-axis coordinates, one-element receiver tuple returns, first-order velocity gradients in 2D/3D and source-waveform gradients in 3D, temporal-resampling adjoint semantics, and radius-M six-face boundary storage with internal terminal pressure fields. Scalar exposes no public initial/final states, and 2D sources remain fixed. The nine public signatures match the SHA-256-pinned 5.0.0 wheel. Existing VRZ, VTI and Elastic contract bodies, historical tutorial evidence and legacy anchors are preserved. New source-build example results are a separate documentation deliverable and are not claimed as wheel GPU validation.

## Scalar3D Example chapter

The flat sidebar now places **Example** directly above **Usage** in both languages. It contains an overview, enclosed-anomaly, surface-only-anomaly and undulating-layer experiments, and report/reproduction downloads. The 21 original plots retain their exact pixels, detailed localized captions, keyboard-scrollable mobile regions and full-resolution links. The report PDF is a repagination of the original Chinese HTML report. Numerical consistency, interior-only directional checks, fixed-model objective, training-pass averages, gradient normalization, and incomplete surface-only recovery remain explicitly distinguished.

Run `python tools/check_scalar3d_examples.py _build/html` to verify the asset allowlist, hashes, plots, navigation order, bilingual sections and downloads. The historical 2D tutorials are preserved; the new Example content is 3D-only.

## V13 / 6.0.0 maintenance · 2026-10-09

Installation and compatibility guidance now cover the V13 / 6.0.0 internal-storage update. Scalar2D compact PML, Radius-M boundary storage, compact full history (retaining boundary_buffer, with the full layout preserved for illumination), and PML-transpose corrections are summarized without changing API calls. Elastic full/boundary no longer use L2 shot-group trajectory scheduling; the existing kernel mapping, including 3D forward shot loops, is retained. Source users must rebuild matching native libraries and restart. Existing tutorials and Scalar3D Example assets keep their original versions, hashes, and evidence scopes. No private implementation, helper module, new test report, or general speedup claim is published. The optional checker is pinned to the published 6.0.0 binary wheel and verifies its metadata, two native-library receipt hashes, and all nine signatures. The official PyPI version record and an independent public download match the 16,674,439-byte artifact and its SHA-256. Python 3.10–3.12 host/CPU installation checks and a clean public-PyPI installation passed; none establishes GPU numerical or performance acceptance.
