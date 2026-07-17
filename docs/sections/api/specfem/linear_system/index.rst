.. _linear_system_api:

``specfem::linear_system``
==========================

Utilities for assembling the spectral-element operator into an explicit
linear system (issue #1982). Provides dense element stiffness extraction --
probing the matrix-free element operator with local unit vectors -- for the
3D elastic isotropic medium, plus a scope validator that rejects meshes
outside the supported tag combination (attenuation, Stacey boundaries).

When SPECFEM++ is built with Trilinos (``SPECFEM_ENABLE_TRILINOS=ON``), the
module additionally provides ``DofMap`` -- the per-medium mapping from
SPECFEM++ ``(iglob, icomp)`` degrees of freedom to Tpetra global ids, with
component-blocked layout ``gid = icomp * nglob + iglob`` matching field
storage -- and ``StiffnessAssembler``, which assembles the global stiffness
matrix :math:`K` of one medium as a ``Tpetra::CrsMatrix`` by scattering
batched dense element blocks into a ``Tpetra::CrsGraph`` built from element
connectivity. The assembled operator satisfies :math:`K u =` internal force
:math:`= -\mathrm{accel}` of the matrix-free
``compute_stiffness_interaction`` kernel (before mass division). Assembly is
serial-only in this milestone; the owned/overlap map split in ``DofMap``
keeps the API ready for distributed Export(ADD) assembly.

.. doxygennamespace:: specfem::linear_system
    :members:
