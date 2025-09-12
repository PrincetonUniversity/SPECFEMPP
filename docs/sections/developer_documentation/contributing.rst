.. _contributing:

Contributing to SPECFEM++
=========================


The easiest way to contribute is by starting to `fork the repository
<https://github.com/PrincetonUniversity/SPECFEMPP/fork>`_, make and commit
changes, and `create a pull request
<https://github.com/PrincetonUniversity/SPECFEMPP/compare>`_ from your updated
fork to the ``devel`` branch of the `SPECFEM++ repository
<https://github.com/PrincetonUniversity/SPECFEMPP>`_.


That being said, we do have some basic guidelines for contributing.

Style guidelines
----------------

To ensure that the code is consistent and follows best practices, the code style
is *automatically* enforced using pre-commit hooks, which are configured in the
`.pre-commit-config.yaml` file in the root of the repository. More about this in
the :ref:`style` documentation.


Github Issues and pull requests handling
----------------------------------------

This includes guidelines for managing GitHub issues and pull requests, which is
described in the :ref:`git-workflow` documentation, which goes over the
branching model and the process for making contributions.


*For maintainers*: Continuous integration (CI)
----------------------------------------------

This includes guidelines for the continuous integration (CI) process, which is
described in the :ref:`tests` documentation. The CI process ensures that the
code is tested and validated before it is merged into the ``main`` or ``devel``
branches.
