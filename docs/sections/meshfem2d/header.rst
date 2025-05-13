
Simulation input parameters
===========================

``title``
~~~~~~~~~

Title of the simulation. This is used to identify the simulation in the output files and logs.

:Type: ``string``

.. code-block::
    :caption: Example

    title = My cool simulation

``NPROC``
~~~~~~~~~

Number of MPI processors to use for the simulation. This is used to determine the number of processes to spawn for the simulation.

:Type: ``integer``

.. code-block::
    :caption: Example

    NPROC = 1

.. note::
    SPECFEM++ currently only supports a single MPI processor. Please set this value to 1.


``OUTPUT_FILES``
~~~~~~~~~~~~~~~~

This is the location where the mesher will store all the output files.

:Type: ``Path`` as ``string``

.. code-block::
    :caption: Example

    OUTPUT_FILES = ./OUTPUT_FILES
