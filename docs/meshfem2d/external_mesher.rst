
External Meshing
================

Parameters here describe the paths to different files generated when using an external mesher.

**Parameter Name**: ``mesh_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Path to the mesh file.

**Type**: ``path``

**Parameter Name**: ``nodes_coords_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Path to the file containing the coordinates of the nodes.

**Type**: ``path``

**Parameter Name**: ``materials_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Path to the file containing the materials number for each element.

**Type**: ``path``

**Parameter Name**: ``free_surface_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Path to the file containing the element number describing the free surface.

**Type**: ``path``

**Parameter Name**: ``axial_elements_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    This parameter is not supported in the solver.

**Description**: Path to the file containing the element number for elements on the axis.

**Type**: ``path``

**Parameter Name**: ``absorbing_surface_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    This parameter is not supported in the solver.

**Description**: Path to the file containing the element number for elements on the absorbing surface.

**Type**: ``path``

**Parameter Name**: ``acoustic_forcing_surface_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    This parameter is not supported in the solver.

**Description**: Path to the file containing the element number for elements on the acoustic forcing surface.

**Type**: ``path``

**Parameter Name**: ``absorbing_cpml_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    This parameter is not supported in the solver.

**Description**: Path to the file containing the element number for elements on the absorbing PML surface.

**Type**: ``path``

**Parameter Name**: ``tangential_detection_curve_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    This parameter is not supported in the solver.

**Description**: Path to the file containing the element number for elements on the tangential curve.

**Type**: ``path``
