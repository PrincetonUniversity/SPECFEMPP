# Homogeneous halfspace with force source, small model, no ABC (implicit suite)

Fixture for the implicit-vs-explicit scheme-identity test
(`ImplicitNewmark3D.ReproducesExplicitSchemeAtBetaZero` in
`../../../dim3/implicit_newmark_tests.cpp`): the same simulation is run
once with the explicit `time_marching` solver and once with
`ImplicitNewmarkSolver` in acceleration form at beta 0, gamma 1/2 -- the
explicit central-difference member of the Newmark family -- and the
recorded seismograms are compared sample by sample. The identity is exact
only where the damping matrix is empty, which is why this fixture has no
absorbing boundaries.

Unlike the `Newmark/serial/dim3` fixtures, this is not a trace-regression
fixture: there is no `traces/` reference and it is not listed in a
`tests.yaml`. The reference is the explicit run computed live by the test.

Everything (mesh provenance, database, source, stations) is copied from
`displacement_tests/Newmark/serial/dim3/HomogeneousHalfspaceSmallNoABCForceSource/`;
see that fixture's README for how to regenerate the Fortran reference. The
`Snakefile` here regenerates `database.bin` from the provenance in this
directory (database only -- no trace rules).
