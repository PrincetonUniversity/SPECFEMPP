.. _specfem_api_ellipticity_index:

``specfem::ellipticity``
========================

.. doxygennamespace:: specfem::ellipticity
    :desc-only:

The ``specfem::ellipticity`` namespace provides compile-time ellipsoid
parameters for geodetic computations. Include via
``#include "specfem/ellipticity.hpp"``.

*  :doc:`model <model>`: Enum selecting the reference ellipsoid (WGS-84, Clarke 1866).
*  :doc:`ellipsoid <ellipsoid>`: Struct template providing semi-major/minor axes for a given model.

.. toctree::
    :maxdepth: 1

    model
    ellipsoid
