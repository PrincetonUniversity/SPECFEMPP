.. _linear_system_api:

``specfem::linear_system``
==========================

Utilities for assembling the spectral-element operator into an explicit
linear system. Provides dense element stiffness
extraction -- probing the matrix-free element operator with local unit
vectors -- for the 3D elastic isotropic medium, plus a scope validator that
rejects meshes outside the supported tag combination. The validator has two
scopes (``StiffnessScope``): the strict default admits natural boundary
conditions only, while ``with_stacey`` additionally admits Stacey boundaries
-- valid because the displacement probe runs at zero velocity, where the
Stacey dashpot contributes nothing to :math:`K`; callers opting in must
assemble the damping matrix separately.

When SPECFEM++ is built with Trilinos (``SPECFEM_ENABLE_TRILINOS=ON``), the
module additionally provides ``DofMap`` -- the per-medium mapping from
SPECFEM++ ``(iglob, icomp)`` degrees of freedom to solver global ids, with
component-blocked layout ``gid = icomp * nglob + iglob`` matching field
storage.

``DofMap`` is a composition of two independent halves
(``BasicDofMap<Numbering, Connections>``), split so the linear system is not
welded to one solver library. ``DofNumbering`` holds the SPECFEM++
quantities -- ``nglob``, ``ncomp``, and the ``gid`` layout -- and names no
library type at all, so it compiles without Trilinos. ``TpetraConnections``
holds everything Teuchos/Tpetra: the communicator, the owned (row) and
overlap (column) maps, and sparsity-graph construction. Callers describe a
matrix's structure as a *coupling pattern* -- a replayable sequence of dense
blocks of global dof ids that all couple to one another -- and
``DofMap::build_graph`` turns it into the library's graph; deriving the
per-row allocation bound and inserting the indices are the library's
business, not the assembler's. Moving to another library that builds
compressed-row matrices from a graph therefore means writing a second
``Connections`` class, leaving the numbering, the assemblers, and the solver
untouched.

``StiffnessAssembler`` assembles the global stiffness matrix :math:`K` of one
medium as a ``Tpetra::CrsMatrix`` by scattering batched dense element blocks
into the graph of its coupling pattern (one block per spectral element, since
every dof of an element couples to every other). The assembled operator
satisfies :math:`K u =` internal force :math:`= -\mathrm{accel}` of the
matrix-free ``compute_stiffness_interaction`` kernel (before mass division).
Assembly is serial-only in this milestone; the owned/overlap map split in
``TpetraConnections`` keeps the API ready for distributed Export(ADD)
assembly.

Toward the implicit Newmark solver (issue #1984), the module also assembles
the remaining operators of the equation of motion
:math:`M \ddot{u} + C \dot{u} + K u = f`:

- ``assemble_mass_vector`` returns the lumped diagonal mass :math:`M` as a
  ``Tpetra::Vector`` by driving the production ``initialize_mass_matrix``
  accumulation with :math:`\Delta t = 0`. The Stacey contribution to the
  lumped mass is exactly :math:`(\Delta t / 2)\, C \, \mathbf{1}` (linear in
  :math:`\Delta t`), so it vanishes identically and the result is the pure
  mass on any mesh, including meshes with Stacey boundaries.
- ``DampingAssembler`` assembles the Stacey damping matrix :math:`C` by
  probing the velocity path of the production stiffness kernel: with
  displacement :math:`\equiv 0` and unit velocity :math:`e_c` at every mesh
  point, the kernel returns :math:`-C e_c` per point. Because the Stacey
  traction is pointwise in velocity, :math:`C` is block-diagonal (one
  symmetric positive-semidefinite ``ncomp x ncomp`` block per boundary GLL
  point) and ``ncomp`` kernel launches recover all blocks. The matrix lives
  on a compact graph whose entries all exist in the stiffness graph, so an
  implicit Newmark operator can be summed on :math:`K`'s graph -- in either
  algebraic form: :math:`M/(\beta \Delta t^2) + \gamma/(\beta \Delta t)\, C +
  K` when solving for :math:`u_{n+1}` (displacement form, :math:`\beta > 0`),
  or :math:`M + \gamma \Delta t\, C + \beta \Delta t^2 K` when solving for
  :math:`a_{n+1}` (acceleration form, :math:`\beta \ge 0`). The two differ by
  the positive factor :math:`1/(\beta \Delta t^2)` wherever both are defined.

``linear_solver_smoke_test`` gates the Belos + Ifpack2 toolchain (GMRES with
a RILUK right preconditioner on ``type_real``) ahead of the implicit solver;
MueLu remains unlinked until the float-only cluster installs are revisited.

.. doxygennamespace:: specfem::linear_system
    :members:
