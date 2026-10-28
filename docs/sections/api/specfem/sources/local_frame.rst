Source quantities in a local frame
==================================

A CMTSOLUTION moment tensor and a FORCESOLUTION force are given in the local
frame at the source, which tilts with latitude and longitude on a globe. Both
are stored as read in the local geocentric :math:`(\theta, \phi, r)` frame
(see :doc:`../coordinate_systems/basis_rotation`) inside a
``LocalFrameQuantity`` and rotated into the Cartesian mesh frame at assembly,
once the source position is placed. On a globe the rotation is
``rotation_matrix<CartesianFrame, GeocentricFrame>(theta, phi)`` at the source;
on a flat regional mesh it is the same rotation at :math:`\theta = 0,\
\phi = -\pi/2` (:math:`x` = East, :math:`y` = North, :math:`z` = Up), which
reproduces the regional relabel ``Mxx = Mpp, Myy = Mtt, Mzz = Mrr, Mxy = -Mtp,
Mxz = Mrp, Myz = -Mrt``.

The CMTSOLUTION reader fills the geocentric slots directly
(``(0,0) = Mtt, (1,1) = Mpp, (2,2) = Mrr, (0,1) = Mtp, (0,2) = Mrt,
(1,2) = Mrp``); the FORCESOLUTION reader reorders East/North/Up through
:doc:`../coordinate_systems/east_north_up`.

.. doxygenclass:: specfem::sources::MomentTensor
    :members:

.. doxygenclass:: specfem::sources::ForceVector
    :members:

.. doxygenfunction:: specfem::sources::rotate(const MomentTensor<SourceFrame>&, const specfem::coordinate_systems::BasisMatrix<TargetFrame, SourceFrame, double>&)

.. doxygenfunction:: specfem::sources::rotate(const ForceVector<SourceFrame>&, const specfem::coordinate_systems::BasisMatrix<TargetFrame, SourceFrame, double>&)

.. doxygenclass:: specfem::sources::LocalFrameQuantity
    :members:
