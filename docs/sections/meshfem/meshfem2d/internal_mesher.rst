
Internal Meshing
================

``interfacesfile``
~~~~~~~~~~~~~~~~~~

The file describing the topography of the simulation domain. For more information, see the :ref:`topography_file` section.

Type
    ``path`` as string

.. code-block::
    :caption: Example

    interfacesfile = "topography.dat"


``xmin``
~~~~~~~~

The minimum x-coordinate of the simulation domain.

Type
    ``real``

.. code-block::
    :caption: Example

    xmin = 0.0


``xmax``
~~~~~~~~

The maximum x-coordinate of the simulation domain.

Type
    ``real``

.. code-block::
    :caption: Example

    xmax = 6400.0


``nx``
~~~~~~

The number of elements in the x-direction.

:Type: ``int``

.. code-block::
    :caption: Example

    nx = 64


``STACEY_ABSORBING_CONDITIONS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``True``, Stacey absorbing boundary conditions are used.

Type
    ``logical``

.. code-block::
    :caption: Example

    STACEY_ABSORBING_CONDITIONS = .true.


``absorbbottom``
~~~~~~~~~~~~~~~~


If ``True``, the bottom boundary is an absorbing boundary.

Type
    ``logical``

.. code-block::
    :caption: Example

    absorbbottom = .true.



``absorbtop``
~~~~~~~~~~~~~

If ``True``, the top boundary is an absorbing boundary.

:Type: ``logical``

.. code-block::
    :caption: Example

    absorbtop = .true.


``absorbleft``
~~~~~~~~~~~~~~

If ``True``, the left boundary is an absorbing boundary.

:Type: ``logical``

.. code-block::
    :caption: Example

    absorbleft = .true.


``absorbright``
~~~~~~~~~~~~~~~

If ``True``, the right boundary is an absorbing boundary.

:Type: ``logical``

.. code-block::
    :caption: Example

    absorbright = .true.

``nbregions``
~~~~~~~~~~~~~

The number of regions in the simulation domain.

:Type: ``int``

.. code-block::
    :caption: Example

    nbregions = 1


Describing a region
-------------------

The region is described using a string. And has following parameters:

- ``nxmin``: Integer value describing the x-coordinate of the spectral element at the bottom left corner of the region.
- ``nxmax``: Integer value describing the x-coordinate of the spectral element at the top right corner of the region.
- ``nzmin``: Integer value describing the z-coordinate of the spectral element at the bottom left corner of the region.
- ``nzmax``: Integer value describing the z-coordinate of the spectral element at the top right corner of the region.
- ``material_number``: Integer value describing the type of material in the region. This value references the material number in the ``velocity_model``.

:Type: ``string``

:Format: ``nxmin nxmax nzmin nzmax material_number``

.. code-block::
    :caption: Example

    0 63 0 63 1

.. note::

    The region description(s) must be preceded by the number of regions in the
    simulation domain. For example, if there are 2 regions, the file should look
    like this if you remove comments:

    .. code-block:: bash

        nbregions = 2
        0 63 0 30 1
        0 63 31 63 2
