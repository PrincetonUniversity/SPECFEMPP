.. _linear_system_api:

``specfem::linear_system``
==========================

Utilities for assembling the spectral-element operator into an explicit
linear system (issues #1982, #1984). Provides dense element stiffness
extraction -- probing the matrix-free element operator with local unit
vectors -- for the 3D elastic isotropic medium, plus a scope validator that
rejects meshes outside the supported tag combination. The validator has two
scopes (``StiffnessScope``): the strict default admits natural boundary
conditions only, while ``with_stacey`` additionally admits Stacey boundaries
-- valid because the displacement probe runs at zero velocity, where the
Stacey dashpot contributes nothing to :math:`K`; callers opting in must
assemble the damping matrix separately.

When SPECFEM++ is built with Trilinos (``SPECFEM_ENABLE_TRILINOS=ON``), the
module additionally provides ``SystemLayout`` -- the per-medium bridge from
SPECFEM++ ``(iglob, icomp)`` degrees of freedom onto Tpetra objects. It owns
the numbering (component-blocked layout ``gid = icomp * nglob + iglob``,
matching field storage), the owned/overlap maps, and the sparsity graphs,
and hands out the structural containers the assemblers fill:
``full_matrix()`` on the fully-connected element-connectivity graph,
``block_diagonal_matrix(mask)`` on a compact graph carrying one
``ncomp x ncomp`` block per admitted mesh point, and ``create_vector()``.
It owns structure only -- the matrices come back zero-valued on a
fill-complete graph, and the assemblers supply the values. Because both
graphs come from one numbering, every entry of a block-diagonal matrix also
exists in the full graph by construction, which is what lets the implicit
Newmark operator sum :math:`C` onto :math:`K`'s graph.

``StiffnessAssembler`` assembles the global stiffness matrix :math:`K` of one
medium by scattering batched dense element blocks into the layout's full
matrix. The assembled operator satisfies :math:`K u =` internal force
:math:`= -\mathrm{accel}` of the matrix-free
``compute_stiffness_interaction`` kernel (before mass division). Assembly is
serial-only in this milestone; the owned/overlap map split in
``SystemLayout`` keeps the API ready for distributed Export(ADD) assembly.

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
  point) and ``ncomp`` kernel launches recover all blocks. The matrix comes
  from ``SystemLayout::block_diagonal_matrix`` with the damping-point mask,
  so an implicit Newmark operator :math:`M/(\beta \Delta t^2) +
  \gamma/(\beta \Delta t)\, C + K` can be summed on :math:`K`'s graph.

``linear_solver_smoke_test`` gates the Belos + Ifpack2 toolchain (GMRES with
a RILUK right preconditioner on ``type_real``) ahead of the implicit solver;
MueLu remains unlinked until the float-only cluster installs are revisited.

.. doxygennamespace:: specfem::linear_system
    :members:
