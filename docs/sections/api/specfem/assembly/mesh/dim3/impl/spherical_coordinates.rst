Spherical coordinates for globe setup
====================================

.. doxygenstruct:: specfem::assembly::mesh_impl::SphericalCoordinates
   :members:

The model-templated 3D assembly constructor fills
``mesh.spherical_coordinates`` only for ``Globe3D``. It remains available during
source, receiver, and property setup, then is released before the constructor
returns. Cartesian assemblies never allocate it. Code constructing only an
assembly mesh can explicitly construct ``SphericalCoordinates(mesh)`` and call
``release()`` after its own setup consumers finish.

The cache uses final, deformed physical coordinates, not the reference geometry
used by the model evaluator. Radius remains in metres: the thin database already
applies its planet's radius scale, so no additional ``PlanetConstants`` conversion
is needed. Each GLL point is indexed in compute-element order.

The shared ``geocentric_coordinates::from_cartesian`` conversion returns
colatitude in [0, pi] and longitude in [0, 2*pi). Globe's ``reduce`` also nudges
zero angles by 1e-7; this cache intentionally preserves them. ``atan2`` handles
the axes without perturbing coordinates, allowing accurate Cartesian round trips.
Longitude is conventionally zero at the poles, and both angles are zero at the
origin. Geographic source/receiver resolution remains a separate concern.
