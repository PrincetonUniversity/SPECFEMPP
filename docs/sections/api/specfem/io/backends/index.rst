
.. _io_backends:

``specfem::io_backends``
========================

.. doxygennamespace:: specfem::io_backends
    :desc-only:

SPECFEM++ supports multiple I/O backends for reading and writing simulation data.
Each backend provides File, Group, and Dataset abstractions with consistent interfaces.


.. toctree::
    :maxdepth: 1

    hdf5
    adios2
    npy
    npz
    ascii
