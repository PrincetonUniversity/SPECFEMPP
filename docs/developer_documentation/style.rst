Style
======

Pre-commit
----------

SPECFEM++ uses pre-commit to check style. Pre-commit can be installed inside python virtual environments. There are several methods for creating virtual environments, but I've found poetry tool is great at managing python environments. Especailly in collaborative environments poetry gives easy method for managing consistency of environments between all contributors via :code:`pyproject.toml` and :code:`poetry.lock` files.

Install poetry
~~~~~~~~~~~~~~~

Download and install poetry using the `official instructions <https://python-poetry.org/docs/#installation>`_.

Install SPECFEM++ development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    poetry install

When you run this command one of two things happen,

1. If a correct version of a package already exists within :code:`poetry.lock` file then poetry installs that version. Here the assumption is that a `poetry.lock` already exists, if not then one is created for you (within SPECFEM++ `poetry.lock` should already exist). `poetry.lock` ensures that all developers have consistent environments.

2. If a correct version of a package doesn't exist within :code:`poetry.lock` then poetry will install a correct version and update `poetry.lock` file.

.. note::

    Make sure you commit :code:`poetry.lock` and :code:`pyproject.toml` files upstream to the remote if you add any packages.

.. note::

    It is also recommended that you run :code:`poetry install` every time you pull the develop branch

Using your python/poetry environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're in the SPECFEM++ root directory you should have access to the poetry environment. To run a command within the environment either

1. Explicitly activate the environment using :code:`poetry shell` as shown below

.. code-block:: bash

    poetry shell
    python <script_name>.py

Or

2. Directly run the command within environment using :code:`poetry run`

.. code-block:: bash

    poetry run python <script_name>.py

Pre-commit hooks
~~~~~~~~~~~~~~~~~

Next, we install pre-commit hooks to check style. Pre-commit hooks are defined in :code:`.pre-commit-config.yaml` within SPECFEM++ root directory. More documentation on pre-commit hooks can be found `here <https://pre-commit.com/hooks.html>`_.

To enable the hooks (This only needs to be done when you clone the repo or there is an update to :code:`.pre-commit-config.yaml`)

.. code-block:: bash

    poetry run pre-commit install

After this, pre-commit should run every time you commit a change. Ensuring that coding style is consistent across all developers.

To manually run pre-commit on all files

.. code-block:: bash

    poetry run pre-commit run --all-files
