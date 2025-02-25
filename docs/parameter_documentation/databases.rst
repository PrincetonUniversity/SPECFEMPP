Databases
#########

Databases section defines location of database files.

.. admonition:: Example of databases section

    .. code-block:: yaml

        databases:
            mesh-database: /path/to/mesh_database.bin


Parameter definitions
=====================


.. dropdown:: ``databases``
    :open:

    The databases section defines the location of files to be read by the
    solver.

    :default value: None

    :possible values: [YAML Node]

    .. _database-file-parameter:

    .. dropdown:: ``databases.mesh-database``
        :open:

        Location of the fortran binary database file defining the mesh

        :default value: None

        :possible values: [string]

        .. admonition:: Example mesh-database

            .. code-block:: yaml

                mesh-database: /path/to/mesh_database.bin
