Quality factor table
====================

The attenuation fit is memoized per Q bucket of width
``1 / specfem::constants::Q_RESOLUTION``, as in SPECFEM3D_GLOBE's
``model_attenuation_storage``. The first Q seen in a bucket represents it, so
the number of fits is bounded by the number of occupied buckets rather than
the number of GLL points. The table is independent of dimension and
formulation: 2D fits with the 1/L-normalized routines and no clipping, 3D with
:doc:`globe` and Q clipped to ``specfem::constants::Q_MAXIMUM``.

.. doxygenclass:: specfem::attenuation::QualityFactorTable
    :members:
