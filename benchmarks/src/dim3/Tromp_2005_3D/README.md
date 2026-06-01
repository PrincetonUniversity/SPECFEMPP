# Tromp_2005 — 3D Elastic Extrusion

This benchmark recreates the 2D Tromp_2005 elastic kernel example in 3D by
extruding the model along y. The x and z geometry, material properties, source
depth, receiver depth, time stepping, and adjoint window match the 2D case.
The only extra degree of freedom is the 3D y component.

The workflow is:

1. Generate an internal 3D mesh with a flat free surface.
2. Run a forward elastic simulation with displacement seismograms.
3. Compute adjoint sources from the recorded BXX, BXY, and BXZ traces.
4. Run the combined simulation and plot the y-collapsed X-Z kernel slice.

Model parameters:

- Domain: 200 km in x, 5 km in y, 80 km in z
- Medium: elastic isotropic, rho = 2600 kg/m^3, Vp = 5800 m/s, Vs = 3198.6 m/s
- Source: vertical force at (50 km, 2.5 km, 40 km depth)
- Receiver: displacement station at (150 km, 2.5 km, 40 km depth)
- Time scheme: Newmark, dt = 0.02 s, nstep = 2004, t0 = 8 s

The benchmark writes elastic kernels under OUTPUT_FILES/Kernels/elastic_isotropic
and plots the same six scalar kernels shown by the 2D benchmark.
