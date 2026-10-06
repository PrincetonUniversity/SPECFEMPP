
.. _injection_fk_api:

``specfem::injection::fk``
==========================

.. doxygennamespace:: specfem::injection::fk
    :desc-only:

The FK solver evaluates an incident teleseismic plane wave in a 1-D layered
medium (elastic P-SV, acoustic, and the fluid-solid interface) at arbitrary 3-D
points, returning displacement / velocity / traction (and pressure for fluid
points) as cubic-B-spline coefficients. The device-parallel engine is reached
through the single :cpp:func:`specfem::injection::fk::solve` entry point.


Solver entry point
------------------

.. doxygenfunction:: specfem::injection::fk::solve

.. doxygenenum:: specfem::injection::fk::field_derivative


Input types
-----------

.. doxygenclass:: specfem::injection::fk::LayeredModel
    :members:

.. doxygenstruct:: specfem::injection::fk::AcousticLayer
    :members:

.. doxygenstruct:: specfem::injection::fk::ElasticIsotropicLayer
    :members:

.. doxygenstruct:: specfem::injection::fk::IncidentWave
    :members:

.. doxygenenum:: specfem::injection::fk::incident_wave_type

.. doxygenstruct:: specfem::injection::fk::TimeWindow
    :members:

.. doxygenstruct:: specfem::injection::fk::FkSizes
    :members:

.. doxygenfunction:: specfem::injection::fk::compute_fk_sizes

.. doxygenstruct:: specfem::injection::fk::EvalPoints
    :members:


Result type
-----------

.. doxygenclass:: specfem::injection::fk::FkResult
    :members:


Model file reader
-----------------

.. doxygenfunction:: specfem::injection::fk::read_fk_model_file
