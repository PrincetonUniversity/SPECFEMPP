
Meshing Parameters
==================

**Parameter Name**: ``PARTITIONING_TYPE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    This parameter is only used for MPI simulations and is ignored for serial simulations.

**Description**: The type of mesh partitioning to use for MPI simulations.
    - SCOTCH: ``3``
    - METIS: ``2``
    - INTERNAL (Ascending): ``1``

**Type**: ``int``

**Parameter Name**: ``NGNOD``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Number of control nodes per element.

**Type**: ``int``
    - Supported values: `4` or `9`

**Parameter Name**: ``database_filename``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Name of the database file to store the generated mesh.

**Type**: ``string``

.. note::
    The database filename needs to be same in meshing and solver parameter files.
