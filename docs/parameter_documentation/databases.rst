Databases
#########

Databases section defines location of database files.

.. admonition:: Example of databases section

    .. code-block:: yaml

        databases:
            mesh-database: /path/to/mesh_database.bin


Parameter definitions
=====================

..dropdown:: ``databases``
    :open:

    :default value: None

    :possible values: [YAML Node]

    :documentation: Define databases section


    .. _database-file-parameter:

    .. dropdown:: ``databases.mesh-database``

        :default value: None

        :possible values: [string]

        :documentation: Location of the fortran binary database file defining the mesh
