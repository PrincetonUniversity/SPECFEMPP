
Receiver Parameters
====================

Define meta parameters
----------------------

``use_existing_STATIONS``
~~~~~~~~~~~~~~~~~~~~~~~~~

If set to ``.true.``, the receivers will be places based on an existing STATIONS file.

:Type: ``logical``

.. code-block::
    :caption: Example

    use_existing_STATIONS = .false.


``nreceiversets``
~~~~~~~~~~~~~~~~~

Number of receiver sets.

:Type: ``int``

.. code-block::
    :caption: Example

    nreceiversets = 1


``anglerec``
~~~~~~~~~~~~

Angle to rotate components at receivers

:Type: ``real``

.. code-block::
    :caption: Example

    anglerec = 0.0d0


``rec_normal_to_surface``
~~~~~~~~~~~~~~~~~~~~~~~~~

If set to ``.true.``, the receiver base angle will be set normal to the surface. Requires external mesh and tangential curve file.

:Type: ``logical``

.. code-block::
    :caption: Example

    rec_normal_to_surface = .false.

.. note::
    This paramter is not supported yet in the solver.

Define receiver sets:
---------------------

Next we define each receiver sets using the following parameters:

``nrec``
~~~~~~~~

Number of receivers in this set. The receivers will be placed at equal distances.

Type
    ``int``

.. code-block::
    :caption: Example

    nrec = 10


``xdeb``
~~~~~~~~

X coordinate of the first receiver in this set.

:Type: ``real``

.. code-block::
    :caption: Example

    xdeb = 0.0d0


``zdeb``
~~~~~~~~

Y coordinate of the first receiver in this set.

:Type: ``real``

.. code-block::
    :caption: Example

    zdeb = 0.0d0

``xfin``
~~~~~~~~

X coordinate of the last receiver in this set.

:Type: ``real``

.. code-block::
    :caption: Example

    xfin = 6400.0d0


``zfin``
~~~~~~~~

Z coordinate of the last receiver in this set.

:Type: ``real``

.. code-block::
    :caption: Example

    zfin = 0.0d0


``record_at_surface_same_vertical``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If set to ``.true.``, the receivers will be placed at the surface of the medium. The vertical position of the receivers will be replaces with topography height.

:Type: ``logical``

.. code-block::
    :caption: Example

    record_at_surface_same_vertical = .false.


``stations_filename``
~~~~~~~~~~~~~~~~~~~~~

Name of the STATIONS file to use. if ``use_existing_STATIONS`` is set to ``.true.``, this defines a file to read receiver locations from. If ``use_existing_STATIONS`` is set to ``.false.``, this defines a file to write receiver locations to.

:Type: ``string``

.. code-block::
    :caption: Example

    stations_filename = stations.dat
