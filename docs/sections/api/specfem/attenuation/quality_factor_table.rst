Quality factor table
====================

The attenuation fit is memoized per Q bucket of width
``1 / specfem::constants::Q_RESOLUTION``, as in SPECFEM3D_GLOBE's
``model_attenuation_storage``. The first Q seen in a bucket represents it, so
the number of fits is bounded by the number of occupied buckets rather than
the number of GLL points. The table is independent of dimension: 3D
clips Q to ``specfem::constants::Q_MAXIMUM`` first, as the SPECFEM3D and
SPECFEM3D_GLOBE meshers do, and 2D uses Q as given, as SPECFEM2D does.

.. doxygenclass:: specfem::attenuation::QualityFactorTable
    :members:

One setup fits Qkappa and Qmu with a pair of tables, passed to every container
it recomputes, so that the isotropic and anisotropic containers agree on each
bucket's representative Q. The fit count of the pair is logged at debug level
by each ``recompute()``; a count close to the number of GLL points means the
table is not being hit.

.. doxygenstruct:: specfem::attenuation::QualityFactorTables
    :members:

.. doxygenstruct:: specfem::attenuation::QualityFactorFitTables
    :members:
