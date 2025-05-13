
Meshing Parameters
==================

``PARTITIONING_TYPE``
~~~~~~~~~~~~~~~~~~~~~

The type of mesh partitioning to use for MPI simulations.
    - SCOTCH: ``3``
    - METIS: ``2``
    - INTERNAL (Ascending): ``1``

:Type: ``int``

.. code-block::
    :caption: Example

    PARTITIONING_TYPE = 3

.. note::

    This parameter is only used for MPI simulations and is ignored for serial simulations.


``NGNOD``
~~~~~~~~~

Number of control nodes per element. Supported values: ``4`` and ``9``

:Type: ``int``

.. code-block::
    :caption: Example

    NGNOD = 9


``database_filename``
~~~~~~~~~~~~~~~~~~~~~

Name of the database file to store the generated mesh.

:Type: ``string``

.. code-block::
    :caption: Example

    database_filename = database.bin

.. note::
    The database filename needs to be same in meshing and solver parameter files.
