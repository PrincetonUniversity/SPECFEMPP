receivers
##########

Receivers section defines receiver information required to calculate seismograms.

.. note::

    Please note that the :ref:`stations_file` is generated using SPECFEM2D mesh generator i.e. xmeshfem2d

**Parameter Name** : ``receivers``
-----------------------------------

**default value** : None

**possible values** : [YAML Node]

**documentation** : receiver information required to calculate seismograms.

**Parameter Name** : ``receivers.stations-file``
******************************************************

**default value** : None

**possible values** : [string]

**documentation** : Path to :ref:`stations_file`

**Parameter Name** : ``receivers.angle``
******************************************************

**default value** : None

**possible values** : [float]

**documentation** : Angle to rotate components at receivers

**Parameter Name** : ``receivers.seismogram-type``
******************************************************

**default value** : None

**possible values** : [YAML list]

**documentation** : Type of seismograms to be written.

.. code-block:: yaml

    seismogram-type:
        - velocity
        - displacement

**Parameter Name** : ``receivers.nstep_between_samples``
*********************************************************

**default value** : None

**possible values** : [int]

**documentation** : Number of time steps between sampling the wavefield at station locations for writing seismogram.

.. admoniiton:: Example receivers section

    .. code-block:: yaml

        receivers:
            stations-file: /path/to/stations_file
            angle: 0.0
            seismogram-type:
                - velocity
                - displacement
            nstep_between_samples: 1
