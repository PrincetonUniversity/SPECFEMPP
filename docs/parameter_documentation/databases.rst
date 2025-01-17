Databases
#########

Databases section defines location of database files.

Parameter definitions
=====================

**Parameter name** : ``databases``
-------------------------------

**default value** : None

**possible values** : [YAML Node]

**documentation** : Define databases section

.. _database-file-parameter:

**Parameter name** : ``databases.mesh-database``
******************************************************

**default value**: None

**possible values**: [string]

**documentation**: Location of the fortran binary database file defining the mesh

**Parameter name** : ``databases.sources``
******************************************************

**default value**: None

**possible values**: [string]

**documentation**: Location of source file (yaml) defining the location of sources

.. admonition:: Example of databases section

    .. code-block:: yaml

        databases:
            mesh-database: /path/to/mesh_database.bin
            sources: /path/to/source_file.yaml
