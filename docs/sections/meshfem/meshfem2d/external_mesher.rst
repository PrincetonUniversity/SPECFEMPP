
External Meshing
================

Parameters here describe the paths to different files generated when using an external mesher.

``mesh_file``
~~~~~~~~~~~~~

Path to the mesh file.

Type
    ``path`` as ``string``

.. code-block::
    :caption: Example

    mesh_file = ./DATA/Mesh_canyon/canyon_mesh_file

``nodes_coords_file``
~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the coordinates of the nodes.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    nodes_coords_file = ./DATA/Mesh_canyon/canyon_nodes_coords


``materials_file``
~~~~~~~~~~~~~~~~~~

Path to the file containing the materials number for each element.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    materials_file = ./DATA/Mesh_canyon/canyon_materials_file

``free_surface_file``
~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number describing the free surface.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    free_surface_file = ./DATA/Mesh_canyon/canyon_free_surface_file


``axial_elements_file``
~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the axis.

:Type: ``path``

.. code-block::
    :caption: Example

    axial_elements_file = ./DATA/Mesh_canyon/canyon_axial_elements_file

.. note::
    This parameter is not supported in the solver.


``absorbing_surface_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the absorbing surface.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    absorbing_surface_file = ./DATA/Mesh_canyon/canyon_absorbing_surface_file

.. note::
    This parameter is not supported in the solver.


``acoustic_forcing_surface_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the acoustic forcing surface.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    acoustic_forcing_surface_file = ./DATA/Mesh_canyon/canyon_acoustic_forcing_surface_file

.. note::
    This parameter is not supported in the solver.


``absorbing_cpml_file``
~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the absorbing PML surface.

:Type: ``path``

.. code-block::
    :caption: Example

    absorbing_cpml_file = ./DATA/Mesh_canyon/canyon_absorbing_cpml_file

.. note::
    This parameter is not supported in the solver.


``tangential_detection_curve_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Path to the file containing the element number for elements on the tangential curve.

:Type: ``path`` as ``string``

.. code-block::
    :caption: Example

    tangential_detection_curve_file = ./DATA/Mesh_canyon/canyon_tangential_detection_curve_file

.. note::
    This parameter is not supported in the solver.
