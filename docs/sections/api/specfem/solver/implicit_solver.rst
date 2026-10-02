``specfem::solver::ImplicitNewmarkSolver``
==========================================

Implicit Newmark solver (issue #1984): one Belos GMRES solve per time step
on an assembled operator, with :math:`K`, :math:`C`, and :math:`M` assembled
by :ref:`specfem::linear_system <linear_system_api>` (stiffness probe, Stacey
velocity-path probe, and the :math:`\Delta t = 0` lumped-mass path).
:math:`A` is constant for a fixed time step, so it is assembled and
preconditioned once (Ifpack2 RILUK with zero fill by default, applied as a
right preconditioner); each step is a single warm-started GMRES solve. MueLu
(algebraic multigrid) stays deferred until the float-only cluster Trilinos
installs are revisited.

Two algebraic forms of the update are available, selected by
:cpp:enum:`specfem::solver::NewmarkForm`:

.. list-table::
   :header-rows: 1
   :widths: 20 45 15 20

   * - Form
     - Operator :math:`A`
     - Unknown
     - Valid for
   * - ``displacement``
     - :math:`M / (\beta \Delta t^2) + \gamma / (\beta \Delta t)\, C + K`
     - :math:`u_{n+1}`
     - :math:`\beta > 0`
   * - ``acceleration``
     - :math:`M + \gamma \Delta t\, C + \beta \Delta t^2 K`
     - :math:`a_{n+1}`
     - :math:`\beta \ge 0`

Wherever both are defined they are the same matrix up to the positive factor
:math:`1 / (\beta \Delta t^2)`, so sparsity, conditioning, and GMRES
behaviour are identical; they differ only in which state variable is the
unknown. The displacement form divides every coefficient by :math:`\beta`
and so cannot represent :math:`\beta = 0` -- there, :math:`u_{n+1}` does not
depend on :math:`a_{n+1}` at all and the map the form must invert is
singular. The acceleration form has no such division, which makes
:math:`\beta = 0,\ \gamma = 1/2` a regular member: the operator reduces to
:math:`M + (\Delta t / 2) C` and the scheme *is* the explicit
central-difference update of the production ``time_marching`` solver. That
identity is what
``ImplicitNewmark3D.ReproducesExplicitSchemeAtBetaZero`` asserts, and it is
the integration-level gate on operator construction and time marching.

Run with a large dissipative step
(:cpp:func:`specfem::solver::NewmarkBetaParameters::dissipative` and a
nonzero ``steady_state_tolerance``) the displacement form is intended to act
as a **static solver**, reaching in a few large solves the state an explicit
run approaches only after many small steps. The steady-state criterion is
velocity/acceleration-based on purpose -- a nonzero-net-force source on a
Stacey-truncated box converges to a constant-velocity drift superposed on the
converged deformation, so a displacement increment never vanishes.

.. note::

   The static-solver path has no integration test yet. An earlier test
   compared it against a live run of SPECFEM++'s own explicit solver, which
   could only freeze current behaviour rather than detect a regression shared
   by both solvers; it was removed pending a genuine reference solution.
   Until then the operator is pinned entrywise, for both forms, by the
   ``ImplicitSolver3D`` unit tests, and the time marching is pinned by the
   :math:`\beta = 0` acceleration-form identity described above.

Only available when SPECFEM++ is built with Trilinos
(``SPECFEM_ENABLE_TRILINOS=ON``). Scope of this milestone: serial, dim3,
single-medium elastic isotropic, NGLL = 5, boundaries ``none``,
``acoustic_free_surface``, or ``stacey``.

.. doxygenenum:: specfem::solver::NewmarkForm

.. doxygenstruct:: specfem::solver::NewmarkBetaParameters
    :members:

.. doxygenstruct:: specfem::solver::ImplicitSolverConfig
    :members:

.. doxygenclass:: specfem::solver::ImplicitNewmarkSolver
    :members:
