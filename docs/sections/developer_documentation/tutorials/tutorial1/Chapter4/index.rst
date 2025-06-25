
.. _Chapter4:

Chapter 4: Understanding assembly C++ struct
============================================

Within SPECFEM++, we utilize the abstraction of data containers, to store the data required, and data access functions, to interface with the data containers. The concept of data containers allows us to group items that are related to each other into single data structures, whereas data access functions allow us to group data items that are generally accessed together into container specific data-types, improving cache locality and reducing memory access times. The concept also allows us to separate the data storage from the computation, making the interface more self-consistent, modular, and easier to maintain.

Data Containers
---------------

The assembly divided into a set of data containers, primarily implemeted as C++ structs. These containers store the data required for computation of wavefield evolution. To elaborate, let us consider the data container used to store spatial derivatives of basis functions (:math:`\partial \xi / \partial x`, :math:`\partial \xi / \partial y`, :math:`\partial \gamma / \partial x`, :math:`\partial \gamma / \partial y`).

.. code:: cpp

    #include <Kokkos_Core.hpp>

    struct jacobian_matrix {
    private:
        using ViewType =
            typename Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                    Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                                    ///< type used to
                                                                    ///< store data

    public:
        int nspec;
        int ngllz;
        int ngllx;

        ViewType xix;
        ViewType xiz;
        ViewType gammax;
        ViewType gammaz;
        ViewType jacobian;

        jacobian_matrix(const int nspec, const int ngllz, const int ngllx)
            : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
              xix("xix", nspec, ngllz, ngllx), xiz("xiz", nspec, ngllz, ngllx),
              gammax("gammax", nspec, ngllz, ngllx),
              gammaz("gammaz", nspec, ngllz, ngllx),
              jacobian("jacobian", nspec, ngllz, ngllx) {}
    };

Data Access Functions
---------------------

To interface with the data containers, it would be useful to define a set of functions that can be used to access the data in a consistent manner. For example, the following snippet demonstrates how a simple function signature could look like.

.. code:: cpp

    template <typename IndexType, typename ContainerType, typename PointAccessType>
    KOKKOS_FUNCTION void load_on_device(const IndexType &index, const ContainerType &container,
                                        PointAccessType &point);

Data access functions allow us to group data items that are generally accessed together into container specific data-types, improving cache locality and reducing memory access times.

The following example shows how to define a data access function for loading spatial derivatives from the data container for a given quadrature point. ``point_jacobian_matrix`` is a struct that holds the spatial derivatives at a given point. Since we generally require all the spatial derivatives at a given point, loading them into a single struct improves cache locality.

.. code:: cpp

    #include <Kokkos_Core.hpp>

    struct index {
        int ispec;
        int iz;
        int ix;
    };

    struct point_jacobian_matrix {
        type_real xix;
        type_real xiz;
        type_real gammax;
        type_real gammaz;
        type_real jacobian;
    };

    KOKKOS_FUNCTION void load_on_device(const index &index, const jacobian_matrix &derivatives,
                                        point_jacobian_matrix &point) {
        point.xix = derivatives.xix(index.ispec, index.iz, index.ix);
        point.xiz = derivatives.xiz(index.ispec, index.iz, index.ix);
        point.gammax = derivatives.gammax(index.ispec, index.iz, index.ix);
        point.gammaz = derivatives.gammaz(index.ispec, index.iz, index.ix);
        point.jacobian = derivatives.jacobian(index.ispec, index.iz, index.ix);
    }

Data Containers and Access Functions in SPECFEM++
-------------------------------------------------

.. admonition:: Feature request
    :class: hint

    We need to define data access functions for the following data containers:

    1. Sources
    2. Receivers
    3. Coupled interfaces

    If you'd like to work on this, please see `issue tracker <https://github.com/PrincetonUniversity/SPECFEMPP/issues/110>`_ for more details.

1. :ref:`Assembled mesh information <assembly_mesh>`
2. :ref:`Jacobian matrix <assembly_jacobian_matrix>`
3. :ref:`Material properties <assembly_properties>`
4. :ref:`Wavefield <assembly_fields>`
5. :ref:`Misfit Kernels <assembly_kernels>`
6. :ref:`Coupled Interfaces <assembly_coupled_interfaces>`
7. :ref:`Boundary Conditions <assembly_boundary>`
8. :ref:`Source Information <assembly_sources>`
9. :ref:`Receiver Information <assembly_receivers>`
