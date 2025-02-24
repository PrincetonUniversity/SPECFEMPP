Runtime Setup
#############

Runtime setup section defines the run-time setup of the simulation. If this section is not defined, the simulation will be a serial simulation with a single run.

Parameter definitions
=====================

**Parameter Name** : ``run-setup`` [optional]
---------------------------------------------

**default value** : number-of-processors: 1, number-of-runs: 1

**possible values** : [YAML Node]

**documentation** : Define run-time configuration for your simulation

**Parameter Name** : ``run-setup.number-of-processors`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**default value** : 1

**possible values** : [int]

**documentation** : Number of MPI processes used in the simulation. MPI version is not enabled in this version of the package. number-of-processors == 1

**Parameter Name** : ``run-setup.number-of-runs`` [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**default value** : 1

**possible values** : [int]

**documentation** : Number of runs in this simulation. Only single run implemented in this version of the package. number-of-runs == 1

.. admonition:: Example run-setup section

    .. code-block:: yaml

        run-setup:
            number-of-processors: 1
            number-of-runs: 1
