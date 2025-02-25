Receiver
########

Receivers section defines receiver information required to calculate seismograms.

.. note::

    Please note that the ``stations_file`` is generated using SPECFEM2D mesh
    generator i.e. xmeshfem2d



.. admonition:: Example receivers section

    .. code-block:: yaml

        receivers:
            stations: /path/to/stations_file
            angle: 0.0
            seismogram-type:
                - velocity
                - displacement
            nstep_between_samples: 1


Parameter definitions
---------------------


    .. dropdown:: ``receivers``
        :open:

        Parameter file section that contains the receiver information required to
        calculate seismograms.

        :default value: None

        :possible values: [YAML Node]


        .. dropdown:: ``receivers.stations``

            Path to ``stations_file``.

            :default value: None

            :possible values: [string]

            .. admonition:: Example stations file

                .. code-block:: yaml

                    stations: /path/to/stations_file


        .. dropdown:: ``receivers.angle``

            Angle to rotate components at receivers

            :default value: None

            :possible values: [float]

            .. admonition:: Example angle

                .. code-block:: yaml

                    angle: 0.0


        .. dropdown:: ``receivers.seismogram-type``

            Type of seismograms to be written.

            :default value: None

            :possible values: [YAML list]

            .. rst-class:: center-table

            +-------------------+---------------------------------------+-------------------------------------+
            |  Seismogram       | SPECFEM Par_file ``seismotype`` value | ``receivers.seismogram-type`` value |
            +===================+=======================================+=====================================+
            | Displacement      |                   1                   |   ``displacement``                  |
            +-------------------+---------------------------------------+-------------------------------------+
            | Velocity          |                   2                   |    ``velocity``                     |
            +-------------------+---------------------------------------+-------------------------------------+
            | Acceleration      |                   3                   |     ``acceleration``                |
            +-------------------+---------------------------------------+-------------------------------------+
            | Pressure          |                   4                   |      ``pressure``                   |
            +-------------------+---------------------------------------+-------------------------------------+
            | Displacement Curl |                   5                   |     ✘ Unsupported                   |
            +-------------------+---------------------------------------+-------------------------------------+
            | Fluid Potential   |                   6                   |     ✘ Unsupported                   |
            +-------------------+---------------------------------------+-------------------------------------+

            .. admonition:: Example seismogram-type

              .. code-block:: yaml

                  seismogram-type:
                      - velocity
                      - displacement


        .. dropdown:: ``receivers.nstep_between_samples``

            Number of time steps between sampling the wavefield at station locations
            for writing seismogram.

            :default value: None

            :possible values: [int]

            .. admonition:: Example nstep_between_samples

                .. code-block:: yaml

                    nstep_between_samples: 1
