#include "../../io/layout_oracle.hpp"
#include "../test_fixture/test_fixture.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/execution.hpp"
#include "specfem/io.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>
#include <vector>

template <bool using_simd, typename ExecutionSpace>
using ParallelConfig = specfem::parallel_configuration::default_chunk_config<
    specfem::element::dimension_tag::dim2,
    specfem::datatype::simd<type_real, using_simd>, ExecutionSpace>;

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ExecutionSpace, typename ViewType>
std::enable_if_t<
    std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>, void>
set_kernel_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const type_real offset) {

  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<
      specfem::tags::Tags<specfem::element::dimension_tag::dim2, MediumTag,
                          PropertyTag, using_simd>>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  specfem::execution::for_all(
      "set_to_value", policy,
      [=](const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        PointType point(static_cast<type_real>(index.ispec + offset));
        specfem::assembly::store_on_host(index, point, kernels);
      });

  Kokkos::fence();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ExecutionSpace, typename ViewType>
std::enable_if_t<
    std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>, void>
check_kernel_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const type_real offset) {

  const auto &kernels = assembly.kernels;
  using PointType = specfem::point::kernels<
      specfem::tags::Tags<specfem::element::dimension_tag::dim2, MediumTag,
                          PropertyTag, using_simd>>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "check_to_value", policy,
      [=](const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
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
          typename ExecutionSpace, typename ViewType>
std::enable_if_t<
    std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>, void>
add_value(const ViewType elements,
          specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
              &assembly,
          const type_real offset) {

  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<
      specfem::tags::Tags<specfem::element::dimension_tag::dim2, MediumTag,
                          PropertyTag, using_simd>>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "add_to_value", policy,
      [=](const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        PointType point(static_cast<type_real>(offset));
        specfem::assembly::add_on_host(index, point, kernels);
      });

  Kokkos::fence();
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ExecutionSpace, typename ViewType>
std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>,
                 void>
set_kernel_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const type_real offset) {

  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<
      specfem::tags::Tags<specfem::element::dimension_tag::dim2, MediumTag,
                          PropertyTag, using_simd>>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "set_to_value", policy,
      KOKKOS_LAMBDA(
          const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        PointType point(static_cast<type_real>(index.ispec + offset));
        specfem::assembly::store_on_device(index, point, kernels);
      });

  Kokkos::fence();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ExecutionSpace, typename ViewType>
std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>,
                 void>
check_kernel_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const type_real offset) {

  const int nspec = assembly.mesh.nspec;
  const int ngll = assembly.mesh.element_grid.ngllx;
  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<
      specfem::tags::Tags<specfem::element::dimension_tag::dim2, MediumTag,
                          PropertyTag, using_simd>>;

  Kokkos::View<PointType ***, Kokkos::DefaultExecutionSpace> point_view(
      "point_view", nspec, ngll, ngll);

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultExecutionSpace>(), elements,
      assembly.mesh.element_grid);
  // Iterate over the elements
  specfem::execution::for_all(
      "check_to_value", policy,
      KOKKOS_LAMBDA(
          const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
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

  // Obtain a host-accessible copy of elements. For Kokkos::View this requires
  // a mirror copy; for ElementIndexRange (and similar non-view types) the
  // object is already host-accessible and can be used directly.
  auto host_elements = [&]() {
    if constexpr (Kokkos::is_view<ViewType>::value) {
      return Kokkos::create_mirror_view_and_copy(
          Kokkos::DefaultHostExecutionSpace(), elements);
    } else {
      return elements;
    }
  }();

  specfem::execution::ChunkedDomainIterator host_policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(),
      host_elements, assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "check_to_value", host_policy,
      [=](const typename decltype(host_policy)::base_index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
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
          typename ExecutionSpace, typename ViewType>
std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>,
                 void>
add_value(const ViewType elements,
          specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
              &assembly,
          const type_real offset) {

  const auto &kernels = assembly.kernels;

  using PointType = specfem::point::kernels<
      specfem::tags::Tags<specfem::element::dimension_tag::dim2, MediumTag,
                          PropertyTag, using_simd>>;
  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  // Iterate over the elements
  specfem::execution::for_all(
      "add_to_value", policy,
      KOKKOS_LAMBDA(
          const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
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
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
              PROPERTY_SET(isotropic, anisotropic),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag,
                specfem::element::attenuation_tag::none);
            set_kernel_value<ElementTags::medium_tag, ElementTags::property_tag,
                             false, Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, offset);
          });

      // Check that we are able to access the values stored in the properties
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
              PROPERTY_SET(isotropic, anisotropic),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag,
                specfem::element::attenuation_tag::none);
            check_kernel_value<ElementTags::medium_tag,
                               ElementTags::property_tag, false,
                               Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, offset);
          });

      // Check that we are able to add the values stored in the properties
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
              PROPERTY_SET(isotropic, anisotropic),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag,
                specfem::element::attenuation_tag::none);
            add_value<ElementTags::medium_tag, ElementTags::property_tag, false,
                      Kokkos::DefaultHostExecutionSpace>(elements, assembly,
                                                         offset);
          });

      // Check that we are able to access the values stored in the properties
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
              PROPERTY_SET(isotropic, anisotropic),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag,
                specfem::element::attenuation_tag::none);
            check_kernel_value<ElementTags::medium_tag,
                               ElementTags::property_tag, false,
                               Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, 2 * offset);
          });

      // SIMD access functions

      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
              PROPERTY_SET(isotropic, anisotropic),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag,
                specfem::element::attenuation_tag::none);
            set_kernel_value<ElementTags::medium_tag, ElementTags::property_tag,
                             false, Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, offset);
          });

      // Check that we are able to access the values stored in the properties
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
              PROPERTY_SET(isotropic, anisotropic),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag,
                specfem::element::attenuation_tag::none);
            check_kernel_value<ElementTags::medium_tag,
                               ElementTags::property_tag, false,
                               Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, offset);
          });

      // Check that we are able to add the values stored in the properties
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
              PROPERTY_SET(isotropic, anisotropic),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag,
                specfem::element::attenuation_tag::none);
            add_value<ElementTags::medium_tag, ElementTags::property_tag, false,
                      Kokkos::DefaultHostExecutionSpace>(elements, assembly,
                                                         offset);
          });

      // Check that we are able to access the values stored in the properties
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
              PROPERTY_SET(isotropic, anisotropic),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag,
                specfem::element::attenuation_tag::none);
            check_kernel_value<ElementTags::medium_tag,
                               ElementTags::property_tag, false,
                               Kokkos::DefaultHostExecutionSpace>(
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

TEST_F(Assembly2D, kernels_io_layout) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto assembly = std::get<5>(parameters);

    const std::string temp_io_directory =
        (std::getenv("BUILD_DIR")
             ? std::string(std::getenv("BUILD_DIR"))
             : boost::filesystem::current_path().string()) +
        "/tests/unit-tests/temp_kernels_io";
    boost::filesystem::remove_all(temp_io_directory);
    boost::filesystem::create_directories(temp_io_directory);

    try {
      // Every kernel entry of element ispec is set to ispec + offset
      const type_real offset = 100;
      specfem::tag_dispatch::for_each(
          assembly.kernels.combinations, [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag);
            if (elements.size() == 0)
              return;
            const auto &container = assembly.kernels.template get_container<
                ElementTags::medium_tag, ElementTags::property_tag>();
            const int base = container.element_range.begin_index();
            container.for_each_host_view(
                [&](const auto view, const std::string) {
                  for (int e = 0; e < elements.size(); ++e)
                    for (std::size_t iz = 0; iz < view.extent(1); ++iz)
                      for (std::size_t ix = 0; ix < view.extent(2); ++ix)
                        view(elements(e) - base, iz, ix) = elements(e) + offset;
                });
          });

      specfem::io::kernel_writer<specfem::io_backends::NPY<specfem::io::write>>
          writer(temp_io_directory);
      writer.write(assembly);

      const std::string folder =
          specfem::MPI::format_proc_filename(temp_io_directory + "/Kernels");

      int n_checked = 0;
      specfem::tag_dispatch::for_each(
          assembly.kernels.combinations, [&]<typename ElementTags>() {
            constexpr auto medium_tag = ElementTags::medium_tag;
            constexpr auto property_tag = ElementTags::property_tag;
            const auto element_range =
                assembly.element_types.get_elements_on_host(medium_tag,
                                                            property_tag);
            const int nelem = element_range.size();
            if (nelem == 0)
              return;
            std::cout << "  kernels_io_layout: "
                      << specfem::element::to_string(medium_tag, property_tag)
                      << " nelem=" << nelem << " (nelem % chunk = "
                      << nelem %
                             specfem::parallel_configuration::storage_chunk_size
                      << ")\n";
            const std::string group =
                folder + "/" +
                specfem::element::to_string(medium_tag, property_tag) + "/";
            const std::size_t ngllz = assembly.mesh.element_grid.ngllz;
            const std::size_t ngllx = assembly.mesh.element_grid.ngllx;

            auto check_shape = [&](const auto &raw, const std::string &n) {
              ASSERT_EQ(raw.shape.size(), 3u) << n;
              ASSERT_EQ(raw.shape[0], static_cast<std::size_t>(nelem)) << n;
              ASSERT_EQ(raw.shape[1], ngllz) << n;
              ASSERT_EQ(raw.shape[2], ngllx) << n;
              ASSERT_FALSE(raw.fortran_order) << n;
            };

            // Coordinates
            for (int c = 0; c < 2; ++c) {
              const std::string n = c == 0 ? "X" : "Z";
              const auto raw =
                  specfem_test::layout_oracle::read_npy_raw<type_real>(
                      group + n + ".npy");
              check_shape(raw, n);
              for (int e = 0; e < nelem; ++e)
                for (std::size_t iz = 0; iz < ngllz; ++iz)
                  for (std::size_t ix = 0; ix < ngllx; ++ix)
                    EXPECT_EQ(
                        raw.at({ std::size_t(e), iz, ix }),
                        assembly.mesh.h_coord(c, element_range(e), iz, ix))
                        << n << " e=" << e;
            }

            // Kernels
            const auto &container =
                assembly.kernels
                    .template get_container<medium_tag, property_tag>();
            container.for_each_host_view(
                [&](const auto, const std::string view_name) {
                  const auto raw =
                      specfem_test::layout_oracle::read_npy_raw<type_real>(
                          group + view_name + ".npy");
                  check_shape(raw, view_name);
                  for (int e = 0; e < nelem; ++e)
                    for (std::size_t iz = 0; iz < ngllz; ++iz)
                      for (std::size_t ix = 0; ix < ngllx; ++ix)
                        EXPECT_NEAR(raw.at({ std::size_t(e), iz, ix }),
                                    element_range(e) + offset, 1e-4)
                            << view_name << " e=" << e;
                  ++n_checked;
                });
          });
      EXPECT_GT(n_checked, 0);
    } catch (std::exception &e) {
      std::cout << "- Test: " << Test.name << "\n- Error: " << e.what()
                << std::endl;
      ADD_FAILURE();
    }
    boost::filesystem::remove_all(temp_io_directory);
  }
}
