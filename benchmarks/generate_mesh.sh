#!/bin/sh
cd data
mkdir -p OUTPUT_FILES/results
xmeshfem2D -p Par_file
cd ../data_iso
mkdir -p OUTPUT_FILES/results
xmeshfem2D -p Par_file
cd ../data_aniso
mkdir -p OUTPUT_FILES/results
xmeshfem2D -p Par_file
