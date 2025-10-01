# Notes for the elastic SPIN test

This test is a simple elastic SPIN test. This test is used to verify the
collapse of the elastic wave propagation with SPIN to the spin case when the
parameters that control the coupling are set to zero, and a force is only
applied on the elastic components.


**IMPORTANT**: The `database.bin` is generated separately from the seismograms
since the seismograms are generate using the generic elastic code, which are the
same as for the homogeneous elastic case! To generate the seismograms with fortran `specfem2d` please use:

- `Par_file`,

to generate the database use

- `Par_file_database` (`topography.dat` is the same).
