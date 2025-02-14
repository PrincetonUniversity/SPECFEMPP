Style
======

Pre-commit
----------

SPECFEM++ uses pre-commit to check style. Pre-commit can be installed inside python virtual environments. There are several methods for creating virtual environments, but I've found uv tool is great at managing python environments. Especailly in collaborative environments uv gives easy method for managing consistency of environments between all contributors via :code:`pyproject.toml` and :code:`uv.lock` files.

Install uv
~~~~~~~~~~

Download and install uv using the `official instructions <https://docs.astral.sh/uv/getting-started/installation>`_.

Install SPECFEM++ development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    uv sync --extra dev

When you run this command one of two things happen,

1. If a correct version of a package already exists within :code:`uv.lock` file then uv installs that version. Here the assumption is that a `uv.lock` already exists, if not then one is created for you (within SPECFEM++ `uv.lock` should already exist). `uv.lock` ensures that all developers have consistent environments.

2. If a correct version of a package doesn't exist within :code:`uv.lock` then uv will install a correct version and update `uv.lock` file.

.. note::

    Make sure you commit :code:`uv.lock` and :code:`pyproject.toml` files upstream to the remote if you add any packages.

.. note::

    It is also recommended that you run :code:`uv sync --extra dev` every time you pull the develop branch

Using your python/uv environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're in the SPECFEM++ root directory with an IDE with Python support, you should have access to the uv environment directly.
To explicitly activate the environment, you can run the following command

.. code-block:: bash

    source .venv/bin/activate

Pre-commit hooks
~~~~~~~~~~~~~~~~

Next, we install pre-commit hooks to check style. Pre-commit hooks are defined in :code:`.pre-commit-config.yaml` within SPECFEM++ root directory. More documentation on pre-commit hooks can be found `here <https://pre-commit.com/hooks.html>`_.

To enable the hooks (This only needs to be done when you clone the repo or there is an update to :code:`.pre-commit-config.yaml`)

.. code-block:: bash

    uv run pre-commit install

After this, pre-commit should run every time you commit a change. Ensuring that coding style is consistent across all developers.

To manually run pre-commit on all files

.. code-block:: bash

    uv run pre-commit run --all-files
