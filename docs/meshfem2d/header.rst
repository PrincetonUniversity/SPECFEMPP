
Simulation input parameters
===========================

**Parameter name**: ``title``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Title of the simulation

**Type**: ``string``

**Parameter name**: ``NPROC``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    SPECFEM++ currently only supports a single MPI processor. Please set this value to 1.

**Description**: Number of MPI processors

**Type**: ``integer``

**Parameter name**: ``OUTPUT_FILES``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Location to store artifacts generated by the mesher

**Type**: ``Path``
