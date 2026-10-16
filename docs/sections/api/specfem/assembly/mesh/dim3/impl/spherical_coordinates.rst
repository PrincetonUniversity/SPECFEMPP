Spherical coordinates for globe setup
====================================

Spherical coordinates are stored directly in the 3D ``points`` struct as
``CoordViewType::host_mirror_type h_spherical_coord``, using ``type_real`` like
the Cartesian and reference coordinate views. No device view is allocated.

The model-templated 3D assembly constructor calls
``mesh.compute_spherical_coordinates()`` only for ``Globe3D``. The view remains
available during source, receiver, and property setup, then is reset before the
constructor returns. Cartesian assemblies never allocate it. Code constructing
only an assembly mesh can call ``mesh.compute_spherical_coordinates()`` and
reset ``mesh.h_spherical_coord = {}`` after its own setup consumers finish.
Consumers must not retain view copies if the allocation is to be freed then.

The cache uses final, deformed physical coordinates, not the reference geometry
used by the model evaluator. Radius remains in metres: the thin database already
applies its planet's radius scale, so no additional ``PlanetConstants`` conversion
is needed. Each GLL point is indexed in compute-element order.

The shared ``cartesian.to_geocentric()`` projection returns colatitude in
[0, pi] and wraps longitude using globe's ``reduce()`` convention. This includes
nudging near-zero angles by 1e-7 radians, so Cartesian round trips near the axes
have a corresponding angular tolerance. Geographic source/receiver resolution
remains a separate concern.
