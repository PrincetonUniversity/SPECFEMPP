# Python Bindings

**Files:** `python/specfempp_core/`

A Python package `specfempp_core` exposes C++ simulation components to Python via nanobind (or pybind11). This enables configuration and execution of SPECFEM++ simulations from Python without writing YAML files manually.

## Enabling

```bash
cmake --preset serial -DSPECFEM_BINDING_PYTHON=ON -B build
cmake --build build
```

## Higher-Level Python API

The [specfempp-py](https://github.com/PrincetonUniversity/SPECFEMPP-py) package provides a higher-level Python API on top of these bindings, wrapping the C++ configuration objects in a Pythonic interface.

---

← [Back to Index](../index.md)
