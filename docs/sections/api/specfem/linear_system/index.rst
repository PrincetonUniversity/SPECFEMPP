.. _linear_system_api:

``specfem::linear_system``
==========================

Utilities for assembling the spectral-element operator into an explicit
linear system (issue #1982). Currently provides dense element stiffness
extraction -- probing the matrix-free element operator with local unit
vectors -- for the 3D elastic isotropic medium, plus a scope validator that
rejects meshes outside the supported tag combination (attenuation, Stacey
boundaries).

.. doxygennamespace:: specfem::linear_system
    :members:
