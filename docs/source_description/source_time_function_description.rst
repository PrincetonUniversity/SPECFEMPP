.. _source_time_function_description:

Source Time Function Description
--------------------------------

.. _dirac_source_description:

Dirac Source Time Function Description
======================================

**Parameter Name** : ``Dirac``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of Dirac source time function

**Parameter Name** : ``Dirac.factor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Scaling factor for Dirac source time function

**Parameter Name** : ``Dirac.tshift``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : 0.0

**possible values** : [float]

**Description** : Time shift for Dirac source time function

.. admonition:: Example

    .. code-block:: yaml

        Dirac:
            factor: 1e10
            tshift: 0.0

.. _ricker_source_description:

Ricker Source Time Function Description
=======================================

**Parameter Name** : ``Ricker``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of Ricker source time function

**Parameter Name** : ``Ricker.factor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Scaling factor for Ricker source time function

**Parameter Name** : ``Ricker.tshift``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : 0.0

**possible values** : [float]

**Description** : Time shift for Ricker source time function

**Parameter Name** : ``Ricker.f0``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Central frequency for Ricker source time function

.. admonition:: Example

    .. code-block:: yaml

        Ricker:
            factor: 1e10
            tshift: 0.0
            f0: 1.0

.. _dgaussian_source_description:

Gaussian Derivative Source Time Function Description
====================================================

**Parameter Name** : ``dGaussian``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of first derivative of a Gaussian source time function

**Parameter Name** : ``dGaussian.factor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Scaling factor for first derivative of a Gaussian time function

**Parameter Name** : ``dGaussian.tshift``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : 0.0

**possible values** : [float]

**Description** : Time shift for first derivative of a Gaussian time function

**Parameter Name** : ``dGaussian.f0``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [float]

**Description** : Central frequency for first derivative of a Gaussian time function

.. admonition:: Example

    .. code-block:: yaml

        dGaussian:
            factor: 1e10
            tshift: 0.0
            f0: 1.0

.. _external_source_description:

External Source Time Function Description
=========================================

**Parameter Name** : ``External``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of External source time function

**Parameter Name** : ``External.format``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : ASCII

**possible values** : [ASCII]

**Description** : Format of the external source time function

**Parameter Name** : ``External.stf``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Location of the external source time function files

**Parameter Name** : ``External.stf.X-component`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : ""

**possible values** : [string]

**Description** : Location of time series trace for X-component of the external source time function (if unset the source time function is set to 0)

**Parameter Name** : ``External.stf.Y-component`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : ""

**possible values** : [string]

**Description** : Location of time series trace for Y-component of the external source time function (if unset the source time function is set to 0)

**Parameter Name** : ``External.stf.Z-component`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : ""

**possible values** : [string]

**Description** : Location of time series trace for Z-component of the external source time function (if unset the source time function is set to 0)

.. Note::

    Atlease one of the components must be set for the external source time function.

.. admonition:: Example

    .. code-block:: yaml

        External:
            format: ascii
            stf:
                X-component: /path/to/X-component.stf
                Z-component: /path/to/Z-component.stf
