
.. _Chapter4:

Understanding ``specfem::compute::assembly``
============================================

For the purpose of this tutorial, we will need to understand how the data is organized in the assembly and the functions that are available to access that data.

Data Containers
---------------

The assembly divided into a set of data containers, primarily implemeted as C++ structs. These containers store the data required for computation of wavefield evolution in a cache friendly manner. To elaborate, let us consider the data container used to store spatial derivatives of basis functions (:math:`\partial \xi / \partial x`, :math:`\partial \xi / \partial y`, :math:`\partial \gamma / \partial x`, :math:`\partial \gamma / \partial y`).

.. code:: cpp

    #include <Kokkos_Core.hpp>

    struct partial_derivatives {
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

        partial_derivatives(const int nspec, const int ngllz, const int ngllx)
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

Data access functions allow us to group data items that are generally accessed together within a single data structure, improving cache locality and reducing memory access times.

The following example shows how to define a data access function for loading spatial derivatives from the data container for a given index. ``point_partial_derivatives`` is a struct that holds the spatial derivatives at a given point. Since we generally require all the spatial derivatives at a given point, loading them into a single struct improves cache locality.

.. code:: cpp

    #include <Kokkos_Core.hpp>

    struct index {
        int ispec;
        int iz;
        int ix;
    };

    struct point_partial_derivatives {
        type_real xix;
        type_real xiz;
        type_real gammax;
        type_real gammaz;
        type_real jacobian;
    };

    KOKKOS_FUNCTION void load_on_device(const index &index, const partial_derivatives &derivatives,
                                        point_partial_derivatives &point) {
        point.xix = derivatives.xix(index.ispec, index.iz, index.ix);
        point.xiz = derivatives.xiz(index.ispec, index.iz, index.ix);
        point.gammax = derivatives.gammax(index.ispec, index.iz, index.ix);
        point.gammaz = derivatives.gammaz(index.ispec, index.iz, index.ix);
        point.jacobian = derivatives.jacobian(index.ispec, index.iz, index.ix);
    }

Data Containers and Access Functions in SPECFEM++
-------------------------------------------------

1. :ref:`Assembled mesh information <assembly_mesh>`
2. :ref:`Partial derivatives <assembly_partial_derivatives>`
3. :ref:`Material properties <assembly_material_properties>`
4. :ref:`Wavefield <assembly_fields>`
5. :ref:`Misfit Kernels <assembly_kernels>`
6. :ref:`Coupled Interfaces <assembly_coupled_interfaces>`
7. :ref:`Boundary Conditions <assembly_boundary>`
8. :ref:`Source Information <assembly_sources>`
9. :ref:`Receiver Information <assembly_receivers>`
