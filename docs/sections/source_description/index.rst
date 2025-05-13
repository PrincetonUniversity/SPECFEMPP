.. _source_description:

Source Description
##################

The sources are defined using YAML format and can be specified in the
``sources.yaml`` file. The sources are defined in a list format, allowing for
any number and combination of sources. The sources can be defined as
``force``, ``moment-tensor``, or ``adjoint-source``. Each source type has its
own set of parameters that can be specified. The full description of possible
values for each source type is given below.


.. dropdown:: ``sources.yaml``
   :open:

    .. dropdown:: ``number-of-sources``

        Total number of sources in the simulation box.

        :Default value: None

        :Possible values: [int]

    .. dropdown:: ``sources``
        :open:

        Definition of sources. Note that this list can contain
        any number and combination of sources.

        :Default value: None

        :Possible values: [List of YAML Nodes]


        .. dropdown:: ``force``

            Definition of force source

            :Default value: None

            :Possible values: [YAML Node]


            .. dropdown:: ``x``

                X coordinate location of the force source.

                :Default value: None

                :Possible values: [float]



            .. dropdown:: ``z``

                Z coordinate location of the force source.

                :Default value: None

                :Possible values: [float]


            .. dropdown:: ``angle``

                Angle of the force source.

                :Default value: 0.0

                :Possible values: [float]


            .. dropdown:: ``Dirac``

                Definition of Dirac source :ref:`dirac_source_description`

                :Default value: None

                :Possible values: [YAML Node]


            .. dropdown:: ``Ricker``

                Definition of Ricker source :ref:`ricker_source_description`

                :Default value: None

                :Possible values: [YAML Node]


            .. dropdown:: ``dGaussian``

                Definition of first derivative of Gaussian time function :ref:`dgaussian_source_description`

                :Default value: None

                :Possible values: [YAML Node]


            .. dropdown:: ``External``

                Definition of External source :ref:`external_source_description`

                :Default value: None

                :Possible values: [YAML Node]



            .. admonition:: Example

                .. code-block:: yaml

                    force:
                        x: 0.0
                        z: 0.0
                        angle: 0.0
                        Dirac:
                            factor: 1e10
                            tshift: 0.0



        .. dropdown:: ``moment-tensor``

            Definition of moment tensor source

            :Default value: None

            :Possible values: [YAML Node]

            .. dropdown:: ``x``

                X coordinate location of the moment tensor source.

                :Default value: None

                :Possible values: [float]


            .. dropdown:: ``z``

                Z coordinate location of the moment tensor source.

                :Default value: None

                :Possible values: [float]


            .. dropdown:: ``Mxx``

                Mxx moment tensor component.

                :Default value: None

                :Possible values: [float]

            .. dropdown:: ``Mzz``

                Mzz moment tensor component.

                :Default value: None

                :Possible values: [float]

            .. dropdown:: ``Mxz``

                Mxz moment tensor component.

                :Default value: None

                :Possible values: [float]

            .. dropdown:: ``Dirac``

                Definition of Dirac source :ref:`dirac_source_description`

                :Default value: None

                :Possible values: [YAML Node]


            .. dropdown:: ``Ricker``

                Definition of Ricker source :ref:`ricker_source_description`

                :Default value: None

                :Possible values: [YAML Node]


            .. dropdown:: ``dGaussian``

                Definition of first derivative Gaussian time function :ref:`dgaussian_source_description`

                :Default value: None

                :Possible values: [YAML Node]

            .. dropdown:: ``External``

                Definition of External source :ref:`external_source_description`

                :Default value: None

                :Possible values: [YAML Node]


            .. admonition:: Example

                .. code-block:: yaml

                    moment-tensor:
                        x: 0.0
                        z: 0.0
                        Mxx: 1e10
                        Mzz: 1e10
                        Mxz: 0.0
                        Ricker:
                            factor: 1e10
                            tshift: 0.0
                            f0: 1.0

        .. dropdown:: ``adjoint-source``

            Definition of adjoint source

            :Default value: None

            :Possible values: [YAML Node]

            .. dropdown:: ``station_name``

                Name of the station.

                :Default value: None

                :Possible values: [string]

            .. dropdown:: ``network_name``

                Name of the network.

                :Default value: None

                :Possible values: [string]


            .. dropdown:: ``x``

                X coordinate location of the adjoint source.

                :Default value: None

                :Possible values: [float]

            .. dropdown:: ``z``

                Z coordinate location of the adjoint source.

                :Default value: None

                :Possible values: [float]

            .. dropdown:: ``angle``

                Angle of the adjoint source.

                :Default value: 0.0

                :Possible values: [float]


            .. dropdown:: ``Dirac``
                Definition of Dirac source :ref:`dirac_source_description`

                :Default value: None

                :Possible values: [YAML Node]


            .. dropdown:: ``dGaussian``

                Definition of first derivative Gaussian time function :ref:`dgaussian_source_description`

                :Default value: None

                :Possible values: [YAML Node]

            .. dropdown:: ``Ricker``

                Definition of Ricker source :ref:`ricker_source_description`

                :Default value: None

                :Possible values: [YAML Node]

            .. dropdown:: ``External``

                Definition of External source :ref:`external_source_description`

                :Default value: None

                :Possible values: [YAML Node]


            .. admonition:: Example

                .. code-block:: yaml

                    adjoint-source:
                        station_name: AA
                        network_name: S0001
                        x: 0.0
                        z: 0.0
                        angle: 0.0
                        Dirac:
                            factor: 1e10
                            tshift: 0.0


    .. admonition:: Example

            .. code-block:: yaml

                number-of-sources: 2
                sources:
                    - force:
                        x: 0.0
                        z: 0.0
                        angle: 0.0
                        Dirac:
                            factor: 1e10
                            tshift: 0.0
                    - moment-tensor:
                        x: 0.0
                        z: 0.0
                        Mxx: 1e10
                        Mzz: 1e10
                        Mxz: 0.0
                        Ricker:
                            factor: 1e10
                            tshift: 0.0
                            f0: 1.0



.. _source_time_function_description:

Source Time Function Description
################################

.. _dirac_source_description:

Dirac Source Time Function Description
======================================

.. dropdown:: ``Dirac``
    :open:

    Definition of Dirac source time function

    :Default value: None

    :Possible values: [YAML Node]

    .. dropdown:: ``Dirac.factor``

        Scaling factor for Dirac source time function

        :Default value: None

        :Possible values: [float]

    .. dropdown:: ``Dirac.tshift``

        Time shift for Dirac source time function

        :Default value: 0.0

        :Possible values: [float]


    .. admonition:: Example

        .. code-block:: yaml

            Dirac:
                factor: 1e10
                tshift: 0.0

.. _ricker_source_description:

Ricker Source Time Function Description
=======================================

.. dropdown:: ``Ricker``\
    :open:

    Definition of Ricker source time function

    :Default value: None

    :Possible values: [YAML Node]

    .. dropdown:: ``factor``

        Scaling factor for Ricker source time function

        :Default value: None

        :Possible values: [float]

    .. dropdown:: ``tshift``

        Time shift for Ricker source time function

        :Default value: 0.0

        :Possible values: [float]

    .. dropdown:: ``f0``

        Central frequency for Ricker source time function

        :Default value: None

        :Possible values: [float]


.. admonition:: Example

    .. code-block:: yaml

        Ricker:
            factor: 1e10
            tshift: 0.0
            f0: 1.0


.. _dgaussian_source_description:

Gaussian Derivative Source Time Function Description
====================================================

.. dropdown:: ``dGaussian``
    :open:

    Definition of first derivative of Gaussian source time function

    :Default value: None

    :Possible values: [YAML Node]

    .. dropdown:: ``dGaussian.factor``

        Scaling factor for first derivative of Gaussian time function

        :Default value: None

        :Possible values: [float]

    .. dropdown:: ``dGaussian.tshift``

        Time shift for first derivative of Gaussian time function

        :Default value: 0.0

        :Possible values: [float]

    .. dropdown:: ``dGaussian.f0``

        Central frequency for first derivative of Gaussian time function

        :Default value: None

        :Possible values: [float]


.. admonition:: Example

    .. code-block:: yaml

        dGaussian:
            factor: 1e10
            tshift: 0.0
            f0: 1.0

.. _external_source_description:

External Source Time Function Description
=========================================

.. dropdown:: ``External``
    :open:

    Definition of external source time function

    :Default value: None

    :Possible values: [YAML Node]

    .. dropdown:: ``External.format``

        Format of the external source time function

        :Default value: ASCII

        :Possible values: [ASCII]

    .. dropdown:: ``External.stf``

        Location of the external source time function files

        :Default value: None

        :Possible values: [YAML Node]

    .. dropdown:: ``External.stf.X-component`` [optional]

        Location of time series trace for X-component of the external source time function (if unset the source time function is set to 0)

        :Default value: ""

        :Possible values: [string]

    .. dropdown:: ``External.stf.Y-component`` [optional]

        Location of time series trace for Y-component of the external source time function (if unset the source time function is set to 0)

        :Default value: ""

        :Possible values: [string]

    .. dropdown:: ``External.stf.Z-component`` [optional]

        Location of time series trace for Z-component of the external source time function (if unset the source time function is set to 0)

        :Default value: ""

        :Possible values: [string]

.. Note::

    Atlease one of the components must be set for the external source time function.

.. admonition:: Example

    .. code-block:: yaml

        External:
            format: ascii
            stf:
                X-component: /path/to/X-component.stf
                Z-component: /path/to/Z-component.stf
