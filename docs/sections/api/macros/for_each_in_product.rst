.. _for_each_in_product:

FOR_EACH_IN_PRODUCT
===================

The ``FOR_EACH_IN_PRODUCT`` macro is a powerful utility in SPECFEMPP for processing combinations of tags that represent different element dimensions, medium types, property types, and boundary conditions.

Description
-----------

This macro executes a specified operation for every combination in the Cartesian product of the provided tag sequences. It significantly reduces code duplication when implementing functionality across multiple specializations.

Syntax
------

.. code-block:: cpp

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(VALUES...), MEDIUM_TAG(VALUES...), ...),
        OPERATION)

The first argument is a tuple of tag sequences, each defined by a macro that can be one of the following:
- ``DIMENSION_TAG(VALUES...)``: Specifies the element dimensions (e.g., ``DIM2``, ``DIM3``).
- ``MEDIUM_TAG(VALUES...)``: Specifies the medium types (e.g., ``ACOUSTIC``, ``ELASTIC_SH``, ``ELASTIC_PSV``).
- ``PROPERTY_TAG(VALUES...)``: Specifies the property types (e.g., ``ISOTROPIC``, ``ANISOTROPIC``).
- ``BOUNDARY_TAG(VALUES...)``: Specifies the boundary conditions (e.g., ``NONE``, ``STACEY``, ``COMPOSITE_STACEY_DIRICHLET``).


The second argument ``OPERATION`` can be one of three types:

* ``INSTANTIATE`` - For template instantiation
* ``DECLARE`` - For variable declaration
* A code block (with optional ``CAPTURE``)

Usage Patterns
--------------

Template Instantiation with INSTANTIATE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to explicitly instantiate templates for all combinations of the specified tags.

.. code-block:: cpp

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
        MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T),
        PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
        BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY, COMPOSITE_STACEY_DIRICHLET)),
        INSTANTIATE(
            (template void specfem::kokkos_kernels::impl::compute_mass_matrix,
            (_DIMENSION_TAG_, specfem::wavefield::simulation_field::forward, 5,
            _MEDIUM_TAG_, _PROPERTY_TAG_, _BOUNDARY_TAG_),
            (const type_real &, const specfem::assembly::assembly &);)))

This expands to template instantiations for each combination of the specified tags. The placeholders ``_DIMENSION_TAG_``, ``_MEDIUM_TAG_``, etc. are substituted with the actual tags defined in the first argument.

Variable Declaration with DECLARE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to declare variables for each tag combination.

.. code-block:: cpp

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
        MEDIUM_TAG(ACOUSTIC, ELASTIC_PSV)),
        DECLARE((IndexViewType, elements),
                (IndexViewType::HostMirror, h_elements)))

This generates declarations like

.. code-block:: cpp

    IndexViewType dim2_elements_acoustic;
    IndexViewType::HostMirror dim2_h_elements_acoustic;
    IndexViewType dim2_elements_elastic_psv;
    IndexViewType::HostMirror dim2_h_elements_elastic_psv;


Code Execution with Code Blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The third usage pattern executes code blocks for each tag combination:

.. code-block:: cpp

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
        MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T)),
        {
            if constexpr (dimension == _dimension_tag_ && medium == _medium_tag_) {
                impl::divide_mass_matrix<dimension, wavefield, _medium_tag_>(assembly);
            }
        })

Inside the code, the current tags are accessible via references like ``_dimension_tag_`` and ``_medium_tag_``.
Optionally, you can capture existing variables using the ``CAPTURE`` macro:

.. code-block:: cpp

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
        MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC, ELASTIC_PSV_T)),
        CAPTURE(elements, h_elements) {
            // Code that uses elements and h_elements with type-specific logic
            if (_medium_tag_ == medium_tag) {
                return _elements_;
            }
        })

The variables inside the ``CAPTURE`` block are captured by reference as variables ``__elements__`` and ``_h_elements_``.

Summary
-------
All three patterns leverage the same underlying mechanism, generating code for every combination of the specified tags, while maintaining type safety and enabling compile-time optimizations. This approach keeps the codebase maintainable while supporting a wide range of material types and simulation parameters.
