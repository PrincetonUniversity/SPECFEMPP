#include "../test_fixture/test_fixture.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_all.hpp"
#include "specfem_setup.hpp"
#include <gtest/gtest.h>

template <bool using_simd, typename ExecutionSpace>
using ParallelConfig = specfem::parallel_config::default_chunk_config<
    specfem::dimension::type::dim2,
    specfem::datatype::simd<type_real, using_simd>, ExecutionSpace>;

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ViewType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultHostExecutionSpace>,
                 void>
set_kernel_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const type_real offset) {

  constexpr auto dimension = specfem::dimension::type::dim2;

  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  specfem::execution::for_all(
      "set_to_value", policy,
      [=](const specfem::point::index<dimension, using_simd> &index) {
        PointType point(static_cast<type_real>(index.ispec + offset));
        specfem::assembly::store_on_host(index, point, kernels);
      });

  Kokkos::fence();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ViewType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultHostExecutionSpace>,
                 void>
check_kernel_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const type_real offset) {

  constexpr auto dimension = specfem::dimension::type::dim2;

  const auto &kernels = assembly.kernels;
  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "check_to_value", policy,
      [=](const specfem::point::index<dimension, using_simd> &index) {
        using datatype = typename PointType::value_type;
        datatype value(static_cast<datatype>(0.0));
        const auto l_offset = offset;

        if constexpr (using_simd) {
          for (std::size_t i = 0; i < index.number_elements; ++i) {
            value[i] = static_cast<type_real>(index.ispec + l_offset);
          }
        } else {
          value = static_cast<type_real>(index.ispec + l_offset);
        }

        PointType expected(value);
        PointType computed;
        specfem::assembly::load_on_host(index, kernels, computed);

        if (computed != expected) {
          std::ostringstream message;

          message << "\n \t Error in function check_to_value";

          message << "\n \t Error at ispec = " << index.ispec
                  << ", iz = " << index.iz << ", ix = " << index.ix << "\n";
          message << "Expected: " << expected.print();
          message << "Got: " << computed.print();
          throw std::runtime_error(message.str());
        }
      });

  Kokkos::fence();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ViewType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultHostExecutionSpace>,
                 void>
add_value(const ViewType elements,
          specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
          const type_real offset) {

  constexpr auto dimension = specfem::dimension::type::dim2;

  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "add_to_value", policy,
      [=](const specfem::point::index<dimension, using_simd> &index) {
        PointType point(static_cast<type_real>(offset));
        specfem::assembly::add_on_host(index, point, kernels);
      });

  Kokkos::fence();
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ViewType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultExecutionSpace>,
                 void>
set_kernel_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const type_real offset) {

  constexpr auto dimension = specfem::dimension::type::dim2;

  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "set_to_value", policy,
      KOKKOS_LAMBDA(const specfem::point::index<dimension, using_simd> &index) {
        PointType point(static_cast<type_real>(index.ispec + offset));
        specfem::assembly::store_on_device(index, point, kernels);
      });

  Kokkos::fence();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ViewType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultExecutionSpace>,
                 void>
check_kernel_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const type_real offset) {

  constexpr auto dimension = specfem::dimension::type::dim2;

  const int nspec = assembly.mesh.nspec;
  const int ngll = assembly.mesh.element_grid.ngllx;
  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;

  Kokkos::View<PointType ***, Kokkos::DefaultExecutionSpace> point_view(
      "point_view", nspec, ngll, ngll);

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultExecutionSpace>(), elements,
      assembly.mesh.element_grid);
  // Iterate over the elements
  specfem::execution::for_all(
      "check_to_value", policy,
      KOKKOS_LAMBDA(const specfem::point::index<dimension, using_simd> &index) {
        PointType computed;
        specfem::assembly::load_on_device(index, kernels, computed);

        const int ispec = index.ispec;
        const int iz = index.iz;
        const int ix = index.ix;

        point_view(ispec, iz, ix) = computed;
      });

  Kokkos::fence();

  const auto point_view_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultHostExecutionSpace(), point_view);

  const auto host_elements = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultHostExecutionSpace(), elements);

  specfem::execution::ChunkedDomainIterator host_policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(),
      host_elements, assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "check_to_value", host_policy,
      [=](const specfem::point::index<dimension, using_simd> &index) {
        using datatype = typename PointType::value_type;
        datatype value(static_cast<datatype>(0.0));

        if constexpr (using_simd) {
          for (std::size_t i = 0; i < index.number_elements; ++i) {
            value[i] = static_cast<type_real>(index.ispec + offset);
          }
        } else {
          value = static_cast<type_real>(index.ispec + offset);
        }

        PointType expected(value);
        const int ispec = index.ispec;
        const int iz = index.iz;
        const int ix = index.ix;

        if (point_view_host(ispec, iz, ix) != expected) {
          std::ostringstream message;

          message << "\n \t Error in function check_to_value";

          message << "\n \t Error at ispec = " << index.ispec
                  << ", iz = " << index.iz << ", ix = " << index.ix << "\n";
          message << "Expected: " << expected.print();
          message << "Got: " << point_view_host(ispec, iz, ix).print();
          throw std::runtime_error(message.str());
        }
      });

  Kokkos::fence();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ViewType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultExecutionSpace>,
                 void>
add_value(const ViewType elements,
          specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
          const type_real offset) {

  constexpr auto dimension = specfem::dimension::type::dim2;

  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;
  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "add_to_value", policy,
      KOKKOS_LAMBDA(const specfem::point::index<dimension, using_simd> &index) {
        PointType point(static_cast<type_real>(offset));
        specfem::assembly::add_on_device(index, point, kernels);
      });

  Kokkos::fence();
}
#endif

TEST_F(Assembly2D, kernels_access_functions) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);

    try {
      const type_real offset = 10.1; // Random offset to store in the properties
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            set_kernel_value<_medium_tag_, _property_tag_, false>(
                elements, assembly, offset);
          })

      // Check that we are able to access the values stored in the properties
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            check_kernel_value<_medium_tag_, _property_tag_, false>(
                elements, assembly, offset);
          })

      // Check that we are able to add the values stored in the properties
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            add_value<_medium_tag_, _property_tag_, false>(elements, assembly,
                                                           offset);
          })

      // Check that we are able to access the values stored in the properties
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            check_kernel_value<_medium_tag_, _property_tag_, false>(
                elements, assembly, 2 * offset);
          });

      // SIMD access functions

      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            set_kernel_value<_medium_tag_, _property_tag_, false>(
                elements, assembly, offset);
          })

      // Check that we are able to access the values stored in the properties
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            check_kernel_value<_medium_tag_, _property_tag_, false>(
                elements, assembly, offset);
          })

      // Check that we are able to add the values stored in the properties
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            add_value<_medium_tag_, _property_tag_, false>(elements, assembly,
                                                           offset);
          })

      // Check that we are able to access the values stored in the properties
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            check_kernel_value<_medium_tag_, _property_tag_, false>(
                elements, assembly, 2 * offset);
          });

      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m " << Test.name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
    } catch (std::exception &e) {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;31m[FAILED]\033[0m \n"
                << "-------------------------------------------------------\n"
                << "- Test: " << Test.name << "\n"
                << "- Error: " << e.what() << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      ADD_FAILURE();
    }
  }
}
