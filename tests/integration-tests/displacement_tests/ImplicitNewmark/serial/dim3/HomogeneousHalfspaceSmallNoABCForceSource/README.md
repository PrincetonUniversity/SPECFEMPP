# Homogeneous halfspace with force source, small model, no ABC (implicit suite)

Fixture for the implicit-vs-explicit dynamic-equivalence test
(`ImplicitNewmark3D.MatchesExplicitRunOnNaturalBoundaryMesh` in
`../../../dim3/implicit_newmark_tests.cpp`): the same simulation is run
once with the explicit `time_marching` solver and once with
`ImplicitNewmarkSolver` (beta 1/4, gamma 1/2), and the recorded
seismograms are compared between the two runs.

Unlike the `Newmark/serial/dim3` fixtures, this is not a trace-regression
fixture: there is no `traces/` reference and it is not listed in a
`tests.yaml`. The reference is the explicit run computed live by the test.

Everything (mesh provenance, database, source, stations) is copied from
`displacement_tests/Newmark/serial/dim3/HomogeneousHalfspaceSmallNoABCForceSource/`;
see that fixture's README for how to regenerate the Fortran reference. The
`Snakefile` here regenerates `database.bin` from the provenance in this
directory (database only -- no trace rules).
