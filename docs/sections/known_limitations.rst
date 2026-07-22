.. _known_limitations:

Known Limitations
=================

This page documents features that are not yet implemented in SPECFEM++. All
limitations listed here are specific to **3-D simulations**; the corresponding
2-D features are fully supported unless noted otherwise.

.. warning::

   Attempting to use any of the features listed below in a 3-D simulation will
   raise a **runtime exception** and terminate the solver. No partial results
   will be written. Check your parameter files against this list before
   submitting a long job.

.. list-table:: Unimplemented 3-D Features
   :widths: 40 20 40
   :header-rows: 1

   * - Feature
     - Dimension
     - Status

   * - PML / absorbing boundary conditions
     - 3-D
     - Not implemented — raises a runtime error when the mesh database reports
       a non-zero number of PML boundaries. Use Stacey absorbing boundaries
       instead where available.

   * - Poroelastic materials
     - 3-D
     - Not implemented — raises a runtime error when a poroelastic material
       is detected in the mesh. Poroelastic media are fully supported in 2-D.

   * - Anisotropic elastic materials
     - 3-D
     - Not implemented — raises a runtime error when an anisotropic elastic
       domain is detected. Anisotropic elastic media are supported in 2-D.

   * - GLL-level property input/output
     - 3-D
     - Not implemented — covers both reading a GLL model
       (``databases.reader.properties``), including tomographic
       (model-file-based) materials, and writing material properties
       (``databases.writer.properties``). 3-D property writing raises
       ``3D property writing not yet implemented``. Both directions are
       supported in 2-D. Seismogram and kernel outputs are unaffected.

   * - Earth-chunk mesh (``MESH_A_CHUNK_OF_THE_EARTH``)
     - 3-D
     - Not implemented — raises a runtime error when the mesh header
       indicates a chunk-of-the-earth topology.

   * - Geocentric coordinate system (Globe3D)
     - 3-D
     - Not implemented — raises a runtime error when geocentric coordinates
       are requested. Cartesian coordinates work normally.

   * - Wavefield plotter output formats other than ``vtkhdf``
     - 3-D
     - Partially implemented — only the ``vtkhdf`` display format is
       supported for 3-D wavefield visualization. All other formats raise a
       runtime error.

Planned Support
---------------

All of the above features are planned for future releases. No specific release
dates are committed at this time. To request prioritization of a particular
feature, please open an issue on the
`GitHub issue tracker <https://github.com/PrincetonUniversity/specfempp/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=>`_.

If you encounter a runtime error that is not listed here, please
`file a bug report <https://github.com/PrincetonUniversity/specfempp/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=>`_.
