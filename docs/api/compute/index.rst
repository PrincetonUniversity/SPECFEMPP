.. _compute_index::

Compute Data Interface
**********************

The interfaces provided here stores data required to compute mass and stiffness terms at elemental level. Compute struct enables easy transfer of data between host and device. Organizing compute struct into smaller structs allows us to a pass these structs to host and device functions and eliminate the need for global arrays. This improves readability and maintainability.

.. toctree::
    :maxdepth: 1

    compute
    compute_partial_derivatives
    compute_properties
    compute_sources
    compute_receivers
    compute_coupled_interfaces
