.. _assembly_injection_surface:

``specfem::assembly::InjectionSurface``
=======================================

The quadrature points of the wavefield-injection surface that the mesher
declares in the database (``mesh.boundaries.injection_surface``). There is one
entry per (element, injection face, GLL point on that face), holding the
element and GLL indices, a unit normal pointing out of the total-field region,
the surface quadrature weight :math:`dS`, the boundary tag of the point, and
the row of the point handed to the injection plugin. Entries and rows are
grouped by material system.

Where the Stacey boundary absorbs, injection uses exactly the face Stacey keeps
for that element and GLL point. Stacey stores one face per point, so at an
element edge shared by two absorbing faces only that face injects, and a point
whose Stacey face is not an injection face is not injected. The absorbing and
injected terms therefore pair up at every point, so the boundary absorbs only
the outgoing part of the field. The container depends only on the geometry, not on
the plugin or on the injection method; :doc:`specfem::injection
<../../injection/index>` builds the plugin's points from it.

.. doxygenstruct:: specfem::assembly::InjectionSurface
    :members:

Dimension-Specific Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: specfem::assembly::InjectionSurface< specfem::element::dimension_tag::dim3 >
    :members:
