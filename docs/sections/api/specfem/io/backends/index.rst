
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

On-disk layout
--------------

Multi-dimensional datasets are stored so that an external reader such as NumPy,
``h5py`` or ``bpls`` recovers the array from the file alone.

**Storage order.** A ``Kokkos::LayoutRight`` view is stored as row-major data
with its own shape. A ``Kokkos::LayoutLeft`` view of rank 2 or higher depends on
the backend:

* HDF5 and ADIOS2 are row-major, so the dataset is declared with reversed
  extents. A view of shape ``(n0, n1, n2)`` appears as ``(n2, n1, n0)``.
* NPY and NPZ keep the view's shape and set ``fortran_order: True`` in the header.

HDF5 and ADIOS2 files written before this convention declared LayoutLeft datasets
with unreversed extents. The readers reject them with a dimension mismatch, unless
the reversed shape happens to equal the original one.

**Properties and kernels.** Each dataset holds logical row-major values of shape
``(nelem, ngllz, ngllx)`` in 2D and ``(nelem, ngllz, nglly, ngllx)`` in 3D,
independent of the build's SIMD width.

**Wavefield coordinates.** ``/Coordinates/<medium>`` holds ``X``, ``Z`` (and ``Y``
in 3D) with one value per global point, ``element_indices`` with the global element
index of each row of ``mapping``, and ``mapping``, which is row-major with shape
``(n_elements, ngllz[, nglly], ngllx)`` and gives the global point index of each
GLL point.

**Attenuation checkpoints.** The ``Attenuation`` group of a wavefield step holds
raw padded rank-1 storage, not logical arrays, because the memory variables are
tiled over the element axis. These datasets are restart state for the same build
configuration: the group's ``storage_chunk_size`` dataset records the chunk size,
and the reader rejects a file whose value differs from the current build.
