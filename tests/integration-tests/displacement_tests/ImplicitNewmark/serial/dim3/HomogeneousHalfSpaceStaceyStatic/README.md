# Homogeneous halfspace, Heaviside force, Stacey ABC (static-solver fixture)

Fixture for the implicit static-solver test
(`ImplicitNewmark3D.RecreatesExplicitSteadyStateWithLargeSteps` in
`../../../dim3/implicit_newmark_tests.cpp`): a Heaviside step force drives
the domain to steady state, once with the explicit solver at `dt = 0.035`
for 600 steps (`specfem_config.yaml`) and once with the implicit solver at
`dt = 0.7` for 30 steps (`specfem_config_implicit.yaml`), both to the same
final time T = 21 s.

Unlike the `Newmark/serial/dim3` fixtures, this is not a trace-regression
fixture: there is no `traces/` reference and it is not listed in a
`tests.yaml`. The reference is the explicit run computed live by the test.

The mesh provenance, database, and stations are copied from
`displacement_tests/Newmark/serial/dim3/HomogeneousHalfSpaceStacey/` --
same domain, same material (rho 2700, vp 3000, vs 1732), Stacey ABCs on
five faces and a free surface on top. The `Snakefile` here regenerates
`database.bin` from the provenance in this directory (database only -- no
trace rules; the identical mesh makes it byte-identical to the Stacey
fixture's).

Only the source time function differs: a Heaviside step with hdur 4 s
(large enough that the implicit run's `dt = 0.7` does not trigger the
Heaviside `hdur >= 5 dt` clamp, which would silently give the two runs
different forcings). `provenance/fortran/DATA/FORCESOLUTION` is adapted to
match (step source, hdur 4), but Fortran reference traces were never
generated for this fixture.
