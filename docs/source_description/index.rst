.. _source_description:

Source Description
##################

Sources are defined using a YAML file.

Parameter Description
=====================

**Parameter Name** : ``number-of-sources``
------------------------------------------

**dafault value** : None

**possible values** : [int]

**Description** : Total number of sources in the simulation box.

**Parameter Name** : ``sources``
--------------------------------

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of sources

**Parameter Name** : ``sources``
--------------------------------

**dafault value** : None

**possible values** : [List of YAML Nodes]

**Description** : Definition of sources. Each node within the list of YAML node defines a single source.

**Parameter Name** : ``sources.force``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of force source

**Parameter Name** : ``sources.moment_tensor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dafault value** : None

**possible values** : [YAML Node]

**Description** : Definition of moment tensor source

**Parameter Name** : ``sources.<source_type>.x``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [float, double]

**Description** : X location of the source. ``source_type`` can be either ``force`` or ``moment_tensor``.

**Parameter Name** : ``sources.<source_type>.z``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [float, double]

**Description** : Z location of the source. ``source_type`` can be either ``force`` or ``moment_tensor``.

**Parameter Name** : ``sources.<source_type>.source_surf``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [bool]

**Description** : Defines if the source is on the surface. If ``false`` then source is inside the medium. ``source_type`` can be either ``force`` or ``moment_tensor``. *source_surf has not been implemented in this version of the package*

**Parameter Name** : ``sources.<source_type>.vx``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [float]

**Description** : Specify velocity in X-direction for a moving source. ``source_type`` can be either ``force`` or ``moment_tensor``. *Moving sources are not implemented in this package*

**Parameter Name** : ``sources.<source_type>.vz``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [float]

**Description** : Specify velocity in Z-direction for a moving source. ``source_type`` can be either ``force`` or ``moment_tensor``. *Moving sources are not implemented in this package*

**Parameter Name** : ``sources.force.angle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [float]

**Description** : Specify angle of a force source

**Parameter Name** : ``sources.moment_tensor.Mxx``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [float]

**Description** : Mxx component of moment tensor source

**Parameter Name** : ``sources.moment_tensor.Mxz``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [float]

**Description** : Mxz component of moment tensor source

**Parameter Name** : ``sources.moment_tensor.Mzz``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [float]

**Description** : Mzz component of moment tensor source

**Parameter Name** : ``sources.<source_type>.Dirac``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**dafault value** : None

**possible values** : [YAML node]

**Description** : Define a Dirac source time function for the source. ``source_type`` can be either ``force`` or ``moment_tensor``. *Only Dirac source time funciton is implemented in this version of the package*

**Parameter Name** : ``sources.<source_type>.Dirac.factor``
***********************************************************

**dafault value** : None

**possible values** : [float]

**Description** : Specify scaling factor for the source time function.

**Parameter Name** : ``sources.<source_type>.Dirac.factor``
***********************************************************

**dafault value** : None

**possible values** : [float]

**Description** : Specify scaling factor for the source time function.

**Parameter Name** : ``sources.<source_type>.Dirac.tshift``
***********************************************************

**dafault value** : None

**possible values** : [float]

**Description** : Specify the time shift for Dirac source time function. Must be 0 if there is only a single source in the simulation.
