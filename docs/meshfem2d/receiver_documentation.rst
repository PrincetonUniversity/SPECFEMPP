
Receiver Parameters
====================

**Parameter Name**: ``use_existing_STATIONS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: If set to ``.true.``, the receivers will be places based on an existing STATIONS file.

**Type**: ``logical``

**Paramter Name**: ``nreceiversets``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Number of receiver sets.

**Type**: ``int``

**Parameter Name**: ``anglerec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Angle to rotate components at receivers

**Type**: ``real``

**Parameter Name**: ``rec_normal_to_surface``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: If set to ``.true.``, the receiver base angle will be set normal to the surface. Requires external mesh and tangential curve file.

**Type**: ``logical``

.. note::
    This paramter is not supported yet in the solver.

Define receiver sets:
---------------------

Next we define each receiver sets using the following parameters:

**Parameter Name**: ``nrec``
*****************************

**Description**: Number of receivers in this set. The receivers will be placed at equal distances.

**Type**: ``int``

**Parameter Name**: ``xdeb``
*****************************

**Description**: X coordinate of the first receiver in this set.

**Type**: ``real``

**Parameter Name**: ``zdeb``
*****************************

**Description**: Y coordinate of the first receiver in this set.

**Type**: ``real``

**Parameter Name**: ``xfin``
*****************************

**Description**: X coordinate of the last receiver in this set.

**Type**: ``real``

**Parameter Name**: ``zfin``
*****************************

**Description**: Y coordinate of the last receiver in this set.

**Type**: ``real``

**Parameter Name**: ``record_at_surface_same_vertical``
******************************************************

**Description**: If set to ``.true.``, the receivers will be placed at the surface of the medium. The vertical position of the receivers will be replaces with topography height.

**Type**: ``logical``

**Parameter Name**: ``stations_filename``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Name of the STATIONS file to use. if ``use_existing_STATIONS`` is set to ``.true.``, this defines a file to read receiver locations from. If ``use_existing_STATIONS`` is set to ``.false.``, this defines a file to write receiver locations to.

**Type**: ``string``
