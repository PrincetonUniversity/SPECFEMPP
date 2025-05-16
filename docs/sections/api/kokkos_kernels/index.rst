
.. _kokkos_kernels:

Kokkos Kernels
==============

.. note::

    Kernel Definition: We refer to kernel to mean Kokkos kernels, which are
    units of ``Kokkos::parallel_for`` or ``Kokkos::parallel_reduce`` that are
    executed within an execution space. I realize this nomeclature might be
    misleading, considering the term "kernel" is also used in the context of
    misfit kernels. When we refer to misfit kernels, we will explicitly use the
    term "misfit kernel".


.. toctree::
    :maxdepth: 1

    domain_kernels/domain_kernels
    frechet_kernels/frechet_kernels
