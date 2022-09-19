Git development workflow
=========================

At SPECFEM we follow master-develop workflow. The master (main) branch is always a stable working code and is generally synced with the latest release of SPECFEM. The develop branch is a stable code with potentially new features which haven't been released in the latest version of SPECFEM yet. If you are contributing to SPECFEM then issue your pull request against the develop branch.

To contribute to the develop branch first checkout develop branch:

.. code-block:: bash

    git clone git@github.com:SPECFEM/specfem2d_kokkos.git
    git checkout develop

.. note::

    It is also recommended that you run :code:`poetry install` every time you pull the develop branch. Please check :ref:`style section<style>` for more information on poetry.

.. note::

    Please also install pre-commit hooks after you've cloned repo. :code:`poetry run pre-commit install`

Next create a feature branch against develop branch. Please be explicit while naming the feature branch.

.. code-block:: bash

    git checkout -b <name-of-the-feature-brach>

Make your contributions inside the feature branch and then commit push them upstream.

.. code-block:: bash

    git push origin <name-of-the-feature-brach>

Finally, issue a pull-request of your branch against the develop branch. To be able to merge your request you need atleast one approved review.
