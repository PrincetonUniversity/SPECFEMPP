.. _specfem_api_mesh_injection_surface:

``specfem::mesh::InjectionSurface``
===================================

Faces where an incident wavefield is injected, as declared by the mesher. The
3D meshers (``xmeshfem3D`` and ``xdecompose_mesh``) select them per side of
the box: every Stacey face when ``STACEY_AS_INJECTION_BOUNDARY = .true.``, or
the sides set with ``INJECTION_XMIN``, ``INJECTION_XMAX``, ``INJECTION_YMIN``,
``INJECTION_YMAX``, ``INJECTION_BOTTOM`` and ``INJECTION_TOP`` otherwise. Every
selected side must be a Stacey absorbing side.

.. doxygenenum:: specfem::mesh::InjectionSurfaceKind

.. doxygenstruct:: specfem::mesh::InjectionSurface

Dimension-Specific Specializations
----------------------------------

.. doxygenstruct:: specfem::mesh::InjectionSurface< specfem::element::dimension_tag::dim3 >
    :members:
