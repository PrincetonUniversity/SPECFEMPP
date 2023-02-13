Runtime Setup
#############

Runtime setup section defines the run-time setup of the simulation.

Parameter definitions
=====================

**Parameter Name** : ``run-setup``
-----------------------------------

**default value** : None

**possible values** : [YAML Node]

**documentation** : Define run-time configuration for your simulation

**Parameter Name** : ``run-setup.number-of-processors``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**default value** : 1

**possible values** : [int]

**documentation** : Number of MPI processes used in the simulation. MPI version is not enabled in this version of the package. number-of-processors == 1

**Parameter Name** : ``run-setup.number-of-runs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**default value** : 1

**possible values** : [int]

**documentation** : Number of runs in this simulation. Only single run implemented in this version of the package. number-of-runs == 1
