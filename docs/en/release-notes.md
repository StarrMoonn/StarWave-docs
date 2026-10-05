# Release Notes

## 2.0.0 · 2026-10-05

Public release of prebuilt Linux x86_64 wheels. This version improves the fused paths for VTI 2D/3D boundary backward. The public Python API and defaults remain unchanged. The move to version 2.0 does not add new scalar/VRZ capabilities. Changes in floating-point operation order mean that bitwise equality with the previous version should not be required.

Restart the interpreter or notebook kernel after upgrading. This version number does not indicate new validation of numerical behavior on target GPUs, multi-GPU execution, FWI, or performance. The known limitations involving model-extension gradients, VRZ stability under strong contrasts, and VTI `epsilon < delta` still apply.

Use [PyPI 2.0.0](https://pypi.org/project/starwave/2.0.0/) as the public release source for installation requirements and the full release scope.

## First draft of this user manual

Adds installation and WSL guidance, a synthetic scalar example, usage guidance for all three equation types, introductions to FWI/DataParallel/INR, an API index, and an FAQ. The first draft used Sphinx with the Read the Docs theme and could be deployed to GitHub Pages or another static server. The current bilingual edition uses Alabaster.

The first draft of this manual includes no CUDA implementation, Python propagation implementation, private experimental data, or internal records. See [Documentation status](status.md) for tutorials and validation work still to be completed.

## Expanded API reference

Scalar, VRZ, and VTI each have a dedicated reference page using the Sphinx Python domain to present complete signatures, individual parameter types/defaults/shapes/units, return structures, autograd scope, and examples. Descriptions of 54 parameters have been added, and the function-entry anchors from the original API index have been preserved. This is a documentation improvement; it does not change the StarWave 2.0.0 runtime interface or validation status.

## Chinese and English edition

The complete manual is available in Chinese and English, including navigation, search, notices, and all 18 content pages. The Chinese site retains Chinese tutorials and navigation, while the API reference body is English in both versions. Each language has an independent build and search index. Language switching preserves the corresponding page and section, and legacy Chinese links and API anchors remain accessible. API presentation follows Deepwave’s Sphinx/Alabaster organization, adds typed signatures, and retains StarWave’s own return contracts and numerical limitations.
