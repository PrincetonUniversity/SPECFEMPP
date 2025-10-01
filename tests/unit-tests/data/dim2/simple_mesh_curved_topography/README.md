# Simple Mesh with Curved Topography

This is a simple mesh with curved topography. It is used for testing purposes.

## Mesh plot

![Mesh plot](gridfile.pdf)

## Mesh Generation

To generate the mesh, run the following command:

```bash

cd <specfempp root directory>
$MESHFEM_PATH -p Par_File

## To generate the pdf
gunplot tests/unit-tests/data/dim2/simple_mesh_curved_topography/plot_gridfile.gnuplot
cd tests/unit-tests/data/dim2/simple_mesh_curved_topography/
ps2pdf gridfile.ps gridfile.pdf

```
