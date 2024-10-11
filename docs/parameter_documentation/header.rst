Header
######

The header section is used for naming the run.

Parameter definitions
=====================

**Parameter name** : ``header``
-------------------------------

**default value** : None

**possible values** : [YAML Node]

**documentation** : Define header section

**Parameter name** : ``header.title``
*****************************************

**default value**: None

**possible values**: [string]

**documentation**: Brief name for this simulation

**Parameter name** : ``header.description``
*********************************************

**default value**: None

**possible values**: [string]

**documentation**: Detailed description for this run.

.. admonition:: Example header

    .. code-block:: yaml

        header:
            title: Heterogeneous acoustic-elastic medium with 1 acoustic-elastic interface # name for your simulation
            description: |
                Material systems : Elastic domain (1), Acoustic domain (1)
                Interfaces : Acoustic-elastic interface (1)
                Sources : Force source (1)
                Boundary conditions : Neumann BCs on all edges
