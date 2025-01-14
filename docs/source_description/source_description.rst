.. _source_description:

Source Description
------------------

Force Source Description
========================

**Parameter Name** : ``sources.force``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of force source

**Parameter Name** : ``sources.force.x``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : X coordinate location of the force source.

**Parameter Name** : ``sources.force.z``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Z coordinate location of the force source.

**Parameter Name** : ``sources.force.angle`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : 0.0

**possible values** : [float]

**Description** : Angle of the force source.

**Parameter Name** : ``sources.force.Dirac`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of Dirac source :ref:`dirac_source_description`

**Parameter Name** : ``sources.force.Ricker`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of Ricker source :ref:`ricker_source_description`

**Parameter Name** : ``sources.force.dGaussian`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of first derivative of a Gaussian time function :ref:`dgaussian_source_description`


**Parameter Name** : ``sources.force.External`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of External source :ref:`external_source_description`

.. admonition:: Example

    .. code-block:: yaml

        force:
            x: 0.0
            z: 0.0
            angle: 0.0
            Dirac:
                factor: 1e10
                tshift: 0.0

Moment Tensor Source Description
================================

**Parameter Name** : ``sources.moment-tensor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of moment tensor source

**Parameter Name** : ``sources.moment-tensor.x``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : X coordinate location of the moment tensor source.

**Parameter Name** : ``sources.moment-tensor.z``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Z coordinate location of the moment tensor source.

**Parameter Name** : ``sources.moment-tensor.Mxx``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Mxx moment tensor component.

**Parameter Name** : ``sources.moment-tensor.Mzz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Mzz moment tensor component.

**Parameter Name** : ``sources.moment-tensor.Mxz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Mxz moment tensor component.

**Parameter Name** : ``sources.moment-tensor.Dirac`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of Dirac source :ref:`dirac_source_description`

**Parameter Name** : ``sources.moment-tensor.Ricker`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of Ricker source :ref:`ricker_source_description`

**Parameter Name** : ``sources.moment-tensor.dGaussian`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of first derivative Gaussian time function :ref:`dgaussian_source_description`

**Parameter Name** : ``sources.moment-tensor.External`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of External source :ref:`external_source_description`

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

Adjoint Source Description
==========================

**Parameter Name** : ``sources.adjoint-source``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of adjoint source

**Parameter Name** : ``sources.adjoint-source.station_name``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [string]

**Description** : Name of the station.

**Parameter Name** : ``sources.adjoint-source.network_name``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [string]

**Description** : Name of the network.

**Parameter Name** : ``sources.adjoint-source.x``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : X coordinate location of the adjoint source.

**Parameter Name** : ``sources.adjoint-source.z``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Z coordinate location of the adjoint source.

**Parameter Name** : ``sources.adjoint-source.angle`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : 0.0

**possible values** : [float]

**Description** : Angle of the adjoint source.

**Parameter Name** : ``sources.adjoint-source.Dirac`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of Dirac source :ref:`dirac_source_description`

**Parameter Name** : ``sources.adjoint-source.Ricker`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of Ricker source :ref:`ricker_source_description`

**Parameter Name** : ``sources.adjoint-source.External`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of External source :ref:`external_source_description`


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
