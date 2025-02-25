Header
######

The header section is used for naming the run.


.. admonition:: Example header

    .. code-block:: yaml

        header:
            title: Heterogeneous acoustic-elastic medium with 1 acoustic-elastic interface # name for your simulation
            description: |
                Material systems : Elastic domain (1), Acoustic domain (1)
                Interfaces : Acoustic-elastic interface (1)
                Sources : Force source (1)
                Boundary conditions : Neumann BCs on all edges

Parameter definitions
=====================

.. dropdown:: ``header``
    :open:

    Define header section for your simulation. This section is used for naming
    the run. but has no impact on the simulation itself.

    :default value: None

    :possible values: [YAML Node]


    .. dropdown:: ``header.title``

        Brief name for this simulation

        :default value: None

        :possible values: [string]

        .. admonition:: Example title

            .. code-block:: yaml

                title: Heterogeneous acoustic-elastic medium with 1 acoustic-elastic interface

    .. dropdown:: ``header.description``

        Detailed description for this run.

        :default value: None

        :possible values: [string]

        .. admonition:: Example description

            This field supports multi-line strings. Use the pipe character (|)
            followed by a newline to start a new line.

            Example:

            .. code-block:: yaml

                description: |
                    This is a long description
                    that spans multiple lines
                    and is rendered as a single paragraph.
