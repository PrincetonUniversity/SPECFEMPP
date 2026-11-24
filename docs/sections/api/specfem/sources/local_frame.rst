Source quantities in a local frame
==================================

A CMTSOLUTION moment tensor and a FORCESOLUTION force are given in the local
frame at the source, which tilts with latitude and longitude on a globe. The
source holds a ``std::variant`` of the tensor (or force) in the local
geocentric :math:`(\theta, \phi, r)` frame (see
:doc:`../coordinate_systems/basis_rotation`) and in the Cartesian mesh frame.
The readers construct the geocentric alternative; once the source position is
placed, assembly replaces it with its rotation into the Cartesian frame. On a globe the rotation is
``rotation_matrix<CartesianFrame, GeocentricFrame>(theta, phi)`` at the source;
on a flat regional mesh it is the same rotation at :math:`\theta = 0,\
\phi = -\pi/2` (:math:`x` = East, :math:`y` = North, :math:`z` = Up), which
reproduces the regional relabel ``Mxx = Mpp, Myy = Mtt, Mzz = Mrr, Mxy = -Mtp,
Mxz = Mrp, Myz = -Mrt``.

The CMTSOLUTION reader fills the geocentric slots directly
(``(0,0) = Mtt, (1,1) = Mpp, (2,2) = Mrr, (0,1) = Mtp, (0,2) = Mrt,
(1,2) = Mrp``); the FORCESOLUTION reader maps East/North/Up :math:`(E, N, U)` to geocentric
:math:`(f_\theta, f_\phi, f_r) = (-N, E, U)`.

.. doxygenclass:: specfem::sources::MomentTensor
    :members:

.. doxygenclass:: specfem::sources::ForceVector
    :members:

.. doxygenfunction:: specfem::sources::rotate(const MomentTensor<SourceFrame>&, const specfem::coordinate_systems::BasisMatrix<TargetFrame, SourceFrame, double>&)

.. doxygenfunction:: specfem::sources::rotate(const ForceVector<SourceFrame>&, const specfem::coordinate_systems::BasisMatrix<TargetFrame, SourceFrame, double>&)
