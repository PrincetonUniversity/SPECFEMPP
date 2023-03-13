Seismograms
###########

Seismograms section defines the output seismograms format at predefined stations.

.. note::

    Please note that the :ref:`stations_file` is generated using SPECFEM meshgenerator i.e. xmeshfem2d

Parameter definitions
=======================

**Parameter Name** : ``seismogram``
------------------------------------

**default value** : None

**possible values** : [YAML Node]

**documentation** : Define seismogram output configuration and location of :ref:`stations_file`

**Parameter Name** : ``seismogram.stations-file``
--------------------------------------------------

**default value** : None

**possible values** : [string]

**documentation** : Path to :ref:`stations_file`

**Parameter Name** : ``seismogram.angle``
------------------------------------------

**default value** : None

**possible values** : [float]

**documentation** : Angle to rotate components at receivers

**Parameter Name** : ``seismogram.seismogram-type``
----------------------------------------------------

**default value** : None

**possible values** : [List of string]

**documentation** : Type of seismograms to be written. The types can be any of displacement, velocity and acceleration. For example, the snippet below will instantiate a writer for to output displacement and velocity seismogram at every station.

.. code:: yaml

    seismogram-type:
        - velocity
        - displacement

**Parameter Name** : ``seismogram.nstep_between_samples``
----------------------------------------------------------

**default value** : None

**possible values** : [int]

**documentation** : Number of time steps between sampling the wavefield at station locations for writing seismogram.

**Parameter Name** : ``seismogram.seismogram-format``
-----------------------------------------------------

**default value** : None

**possible values** : [string]

**documentation** : Type of seismogram format to be written. The possible formats are ascii.

**Parameter Name** : ``seismogram.output-folder``
-------------------------------------------------

**default value** : Current working directory

**possible values** : [string]

**documentation** : Path to output folder where the seismograms will be saved.
