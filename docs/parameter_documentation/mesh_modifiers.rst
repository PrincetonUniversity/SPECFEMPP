Mesh Modifiers
##############

Mesh modifiers are rules to apply to a mesh loaded in from a database file.

Parameter definitions
=====================

**Parameter name** : ``mesh-modifiers``
---------------------------------------

**default value** : None

**possible values** : [YAML Node]

**documentation** : Define databases section


**Parameter name** : ``mesh-modifiers.subdivisions``
****************************************************

**default value**: None

**possible values**: [YAML list]

**documentation**: Collection of subdivision rules.


**Parameter name** : ``mesh-modifiers.subdivisions.material``
*************************************************************

**default value**: None

**possible values**: [int]

**documentation**: The database index of the material (specified in the meshfem parfile) to be subdivided.


**Parameter name** : ``mesh-modifiers.subdivisions.x``
******************************************************

**default value**: 1

**possible values**: [int] (must be positive)

**documentation**: The number of subdivisions along the x-axis. 1 keeps the axis unmodified.

**Parameter name** : ``mesh-modifiers.subdivisions.z``
******************************************************

**default value**: 1

**possible values**: [int] (must be positive)

**documentation**: The number of subdivisions along the z-axis. 1 keeps the axis unmodified.





.. admonition:: Example of mesh-modifiers section

    .. code-block:: yaml

        mesh-modifiers:
            subdivisions:
              - material: 1
                x: 2
                z: 2
              - material: 2
                x: 3
                z: 3
