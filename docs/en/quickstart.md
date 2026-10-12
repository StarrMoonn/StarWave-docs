# Quickstart: A Synthetic Shot Gather

This page retains the 2D scalar introductory wiring, compatible with the published 7.1.0 baseline. For current source 3D models, unequal spacing and first-order velocity/source gradients, see the {ref}`3D scalar call <scalar-3d-example>`. V17-only [Born](usage.md#born-function) has a separate CPU example and background/scattered return contract; it is not part of 7.1.0, and 8.0.0 is not published.

The goal is to check the connections between the model, source, acquisition geometry, and recorded tensor. The example creates a small 32×32 velocity model, one shot with one source, and 16 receivers. It downloads no data, saves no results, and performs no compilation.

First complete [installation](installation.md) with matching native libraries. V17 import prepares existing native dispatch; the script's explicit prepare call is an optional device check there and remains useful for the V16 wheel. Import never compiles or propagates. Then download {download}`scalar_demo.py <../examples/scalar_demo.py>`.

```bash
python scalar_demo.py --mode forward --device 0
```

If you are using the documentation project's source directory:

```bash
python examples/scalar_demo.py --mode forward --device 0
```

## Core code

```{literalinclude} ../examples/scalar_demo.py
:language: python
:pyobject: acquisition
```

The source signal is a Ricker wavelet explicitly defined in this example. It does not call a nonexistent StarWave wavelet utility. The grid axis order is `[x,z]`, and the acquisition coordinates use the same order.

```{literalinclude} ../examples/scalar_demo.py
:language: python
:pyobject: propagate
```

## How to assess the run

The script expects records of shape `(1, 16, 256)` and checks that they are finite and not identically zero. This shape is the expected value under the input contract. No measured results for this separate 32×32 script are supplied. The original notebook experiments have measured [smoke-test](installation.md) and [gradient](modeling/gradient.md) results. A finite/nonzero smoke check also cannot establish that the solution or gradients are correct.

For loading errors, start with the [FAQ](faq.md). For inversion, keep the same source units, grid, and acquisition setup, and continue to [Introduction to FWI](inversion/fwi.md).
