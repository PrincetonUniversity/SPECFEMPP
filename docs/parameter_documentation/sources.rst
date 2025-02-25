Sources
#######

.. admonition:: Example of databases section

    .. code-block:: yaml

        sources: path/to/sources.yaml


Parameter definitions
=====================

.. dropdown:: ``sources``
    :open:

    Define sources section

    :default value: None

    :possible values: [string, YAML Node]


    .. admonition:: Example sources section

        The sources is a path to a YAML file.

        .. code-block:: yaml

            sources: path/to/sources.yaml

        The sources section is a YAML node that contains the source information

        .. code-block:: yaml

            sources:
              number-of-sources: 1
              sources:
                - force:
                    x : 2500.0
                    z : 2500.0
                    source_surf: false
                    angle : 0.0
                    vx : 0.0
                    vz : 0.0
                    Ricker:
                      factor: 1e10
                      tshift: 0.0
                      f0: 10.0


    .. dropdown:: ``sources.number-of-sources``

        Number of sources in the simulation

        :default value: None

        :possible values: [int]

        .. admonition:: Example number-of-sources

            .. code-block:: yaml

                number-of-sources: 1


    .. dropdown:: ``sources.sources``

        List of sources

        :default value: None

        :possible values: [YAML Node]

        .. admonition:: Example sources

            .. code-block:: yaml

                sources:
                  - force:
                      x : 2500.0
                      z : 2500.0
                      source_surf: false
                      angle : 0.0
                      vx : 0.0
                      vz : 0.0
                      Ricker:
                        factor: 1e10
                        tshift: 0.0
                        f0: 10.0
