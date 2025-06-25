.. _point_container:

POINT_CONTAINER
===============

Description
-----------

The ``POINT_CONTAINER`` macro is a key component in SPECFEMPP that provides an efficient and consistent way to define data containers for physical properties and kernels at quadrature points. It significantly reduces code duplication by automatically generating accessor methods, constructors, operators, and printing functionality for point data.

Syntax
------

.. code-block:: cpp

   POINT_CONTAINER(field1, field2, ..., fieldN)

The macro accepts a variable number of field names that will be used to generate the data container's member functions.

Generated Functions
-------------------

For each field provided to the macro, the following functions are generated:

1. **Accessor Methods**: For each field, the macro generates both const and non-const accessors:

   .. code-block:: cpp

      KOKKOS_INLINE_FUNCTION const value_type field1() const;
      KOKKOS_INLINE_FUNCTION value_type& field1();

2. **Index Operators**: Allow array-like access to the internal data:

   .. code-block:: cpp

      KOKKOS_INLINE_FUNCTION const value_type operator[](const int i) const;
      KOKKOS_INLINE_FUNCTION value_type& operator[](const int i);

3. **Constructors**:

   - Default constructor
   - Variadic template constructor accepting individual values
   - Constructor accepting a pointer to values
   - Constructor accepting a single value to initialize all fields

4. **Equality Operators**:

   - Implementation of `==` and `!=` operators for comparing containers

5. **Print Functions**:

   - Regular printing for scalar values
   - SIMD-aware printing for vectorized data

Behind the Scenes
-----------------

The ``POINT_CONTAINER`` macro is a high-level interface that expands to several other macros:

.. code-block:: cpp

   #define POINT_CONTAINER(...) POINT_DATA_CONTAINER_SEQ(POINT_ARGS(__VA_ARGS__))

It uses Boost Preprocessor library to process the arguments and generate the appropriate code. The macro chain includes:

- ``POINT_ARGS``: Converts variadic arguments to a Boost PP sequence
- ``POINT_DATA_CONTAINER_SEQ``: Processes the sequence to create numbered fields
- ``POINT_VALUE_ACCESSORS``: Generates accessor methods
- ``POINT_CONSTRUCTOR``: Creates constructors
- ``POINT_BOOLEAN_OPERATOR_DEFINITION``: Implements comparison operators
- ``POINT_OPERATOR_DEFINITION``: Implements array-like access
- ``POINT_PRINT`` and ``POINT_PRINT_SIMD``: Implement printing functionality

Example Usage
-------------

Here's an example from the elastic media implementation (``core/specfem/point/properties.hpp``):

.. code-block:: cpp

   template <specfem::dimension::type DimensionTag,
             specfem::element::medium_tag MediumTag, bool UseSIMD>
   struct data_container<
       DimensionTag, MediumTag, specfem::element::property_tag::isotropic, UseSIMD,
       std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
       : public PropertyAccessor<DimensionTag, MediumTag,
                                specfem::element::property_tag::isotropic, UseSIMD> {

     using base_type = PropertyAccessor<DimensionTag, MediumTag,
                                      specfem::element::property_tag::isotropic, UseSIMD>;
     using value_type = typename base_type::value_type;
     using simd = typename base_type::simd;

     POINT_CONTAINER(rho, kappa, mu)

     KOKKOS_INLINE_FUNCTION
     const value_type lambdaplus2mu() const {
       return kappa() + static_cast<value_type>(4.0 / 3.0) * mu();
     }

     KOKKOS_INLINE_FUNCTION
     const value_type rho_vp() const { return rho() * lambdaplus2mu(); }

     KOKKOS_INLINE_FUNCTION
     const value_type rho_vs() const { return rho() * mu(); }

     KOKKOS_INLINE_FUNCTION
     const value_type lambda() const {
       return kappa() - static_cast<value_type>(2.0 / 3.0) * mu();
     }
   };

In this example, the ``POINT_CONTAINER(rho, kappa, mu)`` macro generates all the necessary accessor methods and functions for the density (``rho``), bulk modulus (``kappa``), and shear modulus (``mu``) properties. The class then adds custom methods that build upon these basic properties to compute derived quantities like ``lambdaplus2mu``, ``rho_vp``, ``rho_vs``, and ``lambda``.
