#!/bin/sh
cd data
mkdir -p OUTPUT_FILES/results
xmeshfem2d -p Par_file
cd ../data_aniso
mkdir -p OUTPUT_FILES/results
xmeshfem2d -p Par_file
