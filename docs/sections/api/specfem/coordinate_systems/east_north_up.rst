East/North/Up components
========================

Include ``specfem/coordinate_systems/east_north_up.hpp``. The local
East/North/Up frame is the geocentric :math:`(\theta, \phi, r)` frame of
:doc:`basis_rotation` with its axes relabelled: East :math:`= +\mathbf{e}_\phi`,
North :math:`= -\mathbf{e}_\theta`, Up :math:`= +\mathbf{e}_r`. No separate
frame tag exists for it; a vector given as East/North/Up is reordered into
``GeocentricFrame`` components, :math:`(-N, E, U)`, and rotated from there.
These two functions are the single owner of the North sign flip. The
FORCESOLUTION reader uses the forward conversion; receiver seismogram channels
use the inverse.

.. doxygenstruct:: specfem::coordinate_systems::EastNorthUp
    :members:

.. doxygenfunction:: specfem::coordinate_systems::to_geocentric_components

.. doxygenfunction:: specfem::coordinate_systems::to_east_north_up_components
