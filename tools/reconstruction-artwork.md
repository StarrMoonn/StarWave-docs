# Original reconstruction schematics

These eight bilingual SVGs were drawn for the StarWave manual. They illustrate the pinned implementation, not simulated fields, measured memory bars, benchmark results, or a claim of algorithmic novelty. Source: `tools/draw_reconstruction.py` in the documentation repository.

The diagrams use outlined labels so Chinese, mathematical characters and English render consistently without loading fonts from the network. Accessible SVG titles/descriptions and per-label text remain present, with localized HTML alternatives and captions in the chapter. The label shapes use Noto Sans CJK SC 2.004 (SIL OFL 1.1; the existing manual font license is in `docs/_static/fonts/OFL.txt`) and DejaVu Sans for the few mathematical superscripts not in that font. DejaVu derives from Bitstream Vera; its upstream license is at https://dejavu-fonts.github.io/License.html . No font binary is embedded in these illustrations.

Regeneration requires Python, fonttools, Noto Sans CJK SC and DejaVu Sans. On the standard Linux font layout:

    python tools/draw_reconstruction.py

The generator also accepts `--font` and `--font-index`. Publication uses the checked-in vectors; documentation builds do not need these font tools. After editing, render every figure and review both languages before publication. Do not infer physical thicknesses, numerical values or performance from the schematic geometry.
