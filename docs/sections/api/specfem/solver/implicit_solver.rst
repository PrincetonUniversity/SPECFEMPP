``specfem::solver::ImplicitNewmarkSolver``
==========================================

Implicit Newmark solver (issue #1984): one Belos GMRES solve per time step
on the assembled operator
:math:`A = M / (\beta \Delta t^2) + \gamma / (\beta \Delta t)\, C + K`,
with the operators assembled by :ref:`specfem::linear_system
<linear_system_api>` (stiffness probe, Stacey velocity-path probe, and the
:math:`\Delta t = 0` lumped-mass path). :math:`A` is constant for a fixed
time step, so it is assembled and preconditioned once (Ifpack2 RILUK with
zero fill by default, applied as a right preconditioner); each step is a
single warm-started GMRES solve. MueLu (algebraic multigrid) stays deferred
until the float-only cluster Trilinos installs are revisited.

Run with a large dissipative step
(:cpp:func:`specfem::solver::NewmarkBetaParameters::dissipative` and a
nonzero ``steady_state_tolerance``) the solver acts as a **static solver**:
it recreates an explicit run driven to steady state in a few large solves.
The steady-state criterion is velocity/acceleration-based on purpose -- a
nonzero-net-force source on a Stacey-truncated box converges to a
constant-velocity drift superposed on the converged deformation, so a
displacement increment never vanishes.

Only available when SPECFEM++ is built with Trilinos
(``SPECFEM_ENABLE_TRILINOS=ON``). Scope of this milestone: serial, dim3,
single-medium elastic isotropic, NGLL = 5, boundaries ``none``,
``acoustic_free_surface``, or ``stacey``.

.. doxygenstruct:: specfem::solver::NewmarkBetaParameters
    :members:

.. doxygenstruct:: specfem::solver::ImplicitSolverConfig
    :members:

.. doxygenclass:: specfem::solver::ImplicitNewmarkSolver
    :members:
