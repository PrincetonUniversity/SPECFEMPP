
Internal Meshing
================

**Parameter Name**: ``interfacesfile``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: The file describing the topography of the simulation domain. For more information, see the :ref:`topography_file` section.

**Type**: ``path``

**Parameter Name**: ``xmin``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: The minimum x-coordinate of the simulation domain.

**Type**: ``real``

**Parameter Name**: ``xmax``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: The maximum x-coordinate of the simulation domain.

**Type**: ``real``

**Parameter Name**: ``nx``
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: The number of elements in the x-direction.

**Type**: ``int``

**Parameter Name**: ``absorbbottom``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    Only free surface boundary conditions are currently supported.

**Description**: If ``True``, the bottom boundary is an absorbing boundary.

**Type**: ``logical``

**Parameter Name**: ``absorbtop``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    Only free surface boundary conditions are currently supported.

**Description**: If ``True``, the top boundary is an absorbing boundary.

**Type**: ``logical``

**Parameter Name**: ``absorbleft``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    Only free surface boundary conditions are currently supported.

**Description**: If ``True``, the left boundary is an absorbing boundary.

**Type**: ``logical``

**Parameter Name**: ``absorbright``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    Only free surface boundary conditions are currently supported.

**Description**: If ``True``, the right boundary is an absorbing boundary.

**Type**: ``logical``

**Parameter Name**: ``nbregions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: The number of regions in the simulation domain.

**Type**: ``int``

Describing a region
-------------------

The region is described using a string.

**string value**: ``nxmin nxmax nzmin nzmax material_number``

**Description**:
    - ``nxmin``: Integer value describing the x-coordinate of the spectral element at the bottom left corner of the region.
    - ``nxmax``: Integer value describing the x-coordinate of the spectral element at the top right corner of the region.
    - ``nzmin``: Integer value describing the z-coordinate of the spectral element at the bottom left corner of the region.
    - ``nzmax``: Integer value describing the z-coordinate of the spectral element at the top right corner of the region.
    - ``material_number``: Integer value describing the type of material in the region. This value references the material number in the :ref:`velocity_model` section.
