Style
======

Pre-commit
===========

SPECFEM uses pre-commit to check style. Pre-commit can be installed inside python virtual environments. There are several methods for creating virtual environments, but I've found poetry tool is great at managing python environments. Especailly in collaborative environments poetry gives easy method for managing consistency of environments between all contributors via :code:`pyproject.toml` and :code:`poetry.lock` files. But more on this later.

Here we give a short tutorial for using poetry specifically with intention of using pre-commit hooks, but the information provided here can be translated for managing any python environment. The official poetry documentation can be found at `poetry docs <https://python-poetry.org/docs/>`_

Poetry Installation
~~~~~~~~~~~~~~~~~~~~

Poetry requires python 2.7 or python 3.5+. However, we recommend that you use python 3.7+. First, download and install poetry:

.. code-block:: bash

    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

You can check if installation succeeded using

.. code-block:: bash

    poetry --version

Adding python packages
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    This section is only for educational purposes since SPECFEM comes with a pyproject.toml

We define all the packages required within our environment using the `pyproject.toml` file. For example, if we needed to add pre-commit to our environment we would define it within :code:`pyproject.toml` as

.. code-block:: toml

    [tool.poetry]
    name = "specfem2d_kokkos"
    version = "0.1.0"
    description = "Kokkos implementation of SpecFEM2D code"
    authors = ["Your Name <you@example.com>"]

    [tool.poetry.dependencies]
    python = ">=3.7,<4.0"
    pre-commit = "^=2.19.0"

    [tool.poetry.dev-dependencies]

    [build-system]
    requires = ["poetry-core>=1.0.0"]
    build-backend = "poetry.core.masonry.api"

As stated above we define our python, and pre-commit versions within :code:`pyproject.toml`.

Install python packages
~~~~~~~~~~~~~~~~~~~~~~~~

To install python packages run

.. code-block:: bash

    poetry install

When you run this command two things happen,

1. If a correct version of a package already exists within :code:`poetry.lock` file then poetry installs that version. Here the assumption is that a `poetry.lock` already exists, if not then one is created for you (within SPECFEM `poetry.lock` should already exist). `poetry.lock` ensures that all developers have consistent environments.

2. If a correct version of a package doesn't exist within :code:`poetry.lock` then poetry will install a correct version and update `poetry.lock` file.

.. note::

    Make sure you commit :code:`poetry.lock` and :code:`pyproject.toml` files upstream in the remote if you add any packages.

.. note::

    It is also recommended that you run :code:`poetry install` every time you pull the develop branch
