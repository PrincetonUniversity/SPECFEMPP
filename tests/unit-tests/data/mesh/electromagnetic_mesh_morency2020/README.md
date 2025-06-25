# Electro Magentic Mesh database for Morency 2020

This database contains the electromagnetic mesh used in the following publication:

- Morency, C. "Electromagnetic wave propagation based upon spectral-element
  methodology in dispersive and attenuating media", _Geophysical Journal
  International_, Volume 220, Issue 2, February 2020, Pages 951–966,
  https://doi.org/10.1093/gji/ggz510

The example mesh corresponds to Figures 4 and 5, with Table 4 showing the parameters used for the meshing.

The mesh was created using the SPECFEM2D Fortran code, and the metadata for the mesh is available in the `DATA` directory.

**Note**: The mesh was created using the `run_this_example.sh` script in this
directory. This only works from the specfem fortran directory since it requires
relative paths to the `bin` and `DATA` directories.
