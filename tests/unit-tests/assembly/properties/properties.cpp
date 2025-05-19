#include "../execution_pattern/forall.hpp"
#include "../test_fixture/test_fixture.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/property/reader.hpp"
#include "io/property/writer.hpp"
#include "specfem_setup.hpp"
#include <gtest/gtest.h>

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ViewType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultHostExecutionSpace>,
                 void>
set_value(const ViewType elements, specfem::compute::assembly &assembly,
          const type_real offset) {

  const auto &properties = assembly.properties;

  using PointType =
      specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                 PropertyTag, using_simd>;

  using PolicyType = execution_pattern::PolicyType<ViewType, using_simd>;

  // Iterate over the elements
  execution_pattern::forall<using_simd>(
      "set_to_value", elements, assembly.mesh.ngllx,
      [=](const typename PolicyType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.index;
        PointType point(static_cast<type_real>(index.ispec + offset));
        specfem::compute::store_on_host(index, point, properties);
      });

  Kokkos::fence();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ViewType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultHostExecutionSpace>,
                 void>
check_value(const ViewType elements, specfem::compute::assembly &assembly,
            const type_real offset) {
  const auto &properties = assembly.properties;
  using PointType =
      specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                 PropertyTag, using_simd>;

  using PolicyType = execution_pattern::PolicyType<ViewType, using_simd>;

  // Iterate over the elements
  execution_pattern::forall<using_simd>(
      "check_to_value", elements, assembly.mesh.ngllx,
      [=](const typename PolicyType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.index;
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

        PointType computed;
        specfem::compute::load_on_host(index, properties, computed);

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

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ViewType>
std::enable_if_t<std::is_same_v<typename ViewType::execution_space,
                                Kokkos::DefaultExecutionSpace>,
                 void>
check_value(const ViewType elements, specfem::compute::assembly &assembly,
            const type_real offset) {

  const int nspec = assembly.mesh.nspec;
  const int ngll = assembly.mesh.ngllx;
  const auto &properties = assembly.properties;

  using PointType =
      specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                 PropertyTag, using_simd>;

  Kokkos::View<PointType ***, Kokkos::DefaultExecutionSpace> point_view(
      "point_view", nspec, ngll, ngll);

  // Iterate over the elements
  execution_pattern::forall<using_simd>(
      "check_to_value", elements, assembly.mesh.ngllx,
      KOKKOS_LAMBDA(
          const typename execution_pattern::PolicyType<ViewType, using_simd>::
              iterator_type::index_type &iterator_index) {
        const auto index = iterator_index.index;
        PointType computed;
        specfem::compute::load_on_device(index, properties, computed);

        const int ispec = index.ispec;
        const int iz = index.iz;
        const int ix = index.ix;

        point_view(ispec, iz, ix) = computed;
      });

  Kokkos::fence();
  const auto point_view_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultHostExecutionSpace(), point_view);

  // const auto host_elements = Kokkos::create_mirror_view_and_copy(
  //     Kokkos::DefaultHostExecutionSpace(), elements);
  // using HostPolicyType =
  //     execution_pattern::PolicyType<decltype(host_elements), using_simd>;

  // execution_pattern::forall<using_simd>(
  //     "check_to_value", host_elements, assembly.mesh.ngllx,
  //     [=](const typename HostPolicyType::iterator_type::index_type
  //             &iterator_index) {
  //       const auto index = iterator_index.index;
  //       PointType expected(static_cast<type_real>(index.ispec + offset));
  //       const int ispec = index.ispec;
  //       const int iz = index.iz;
  //       const int ix = index.ix;

  //       if (point_view_host(ispec, iz, ix) != expected) {
  //         std::ostringstream message;

  //         message << "\n \t Error in function check_to_value";

  //         message << "\n \t Error at ispec = " << index.ispec
  //                 << ", iz = " << index.iz << ", ix = " << index.ix << "\n";
  //         message << "Expected: " << expected.print();
  //         message << "Got: " << point_view_host(ispec, iz, ix).print();
  //         throw std::runtime_error(message.str());
  //       }
  //     });

  Kokkos::fence();
}
#endif

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void check_compute_to_mesh(
    const specfem::compute::assembly &assembly,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh) {

  const int ngll = assembly.mesh.ngllx;
  const auto &properties = assembly.properties;
  const auto &element_types = assembly.element_types;
  const auto &mapping = assembly.mesh.mapping;
  const auto &materials = mesh.materials;

  // Get all elements of the given type
  const auto elements =
      element_types.get_elements_on_host(MediumTag, PropertyTag);

  using PointType = specfem::point::properties<specfem::dimension::type::dim2,
                                               MediumTag, PropertyTag, false>;

  using PolicyType = execution_pattern::PolicyType<decltype(elements), false>;

  // Iterate over the elements
  execution_pattern::forall<false>(
      "check_compute_to_mesh", elements, ngll,
      [=](const typename PolicyType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.index;
        const int ispec = index.ispec;

        // Get the properties stored within the mesh
        const int ispec_mesh = mapping.compute_to_mesh(ispec);
        const auto expected =
            materials.get_material<MediumTag, PropertyTag>(ispec_mesh)
                .get_properties();

        // Get the properties stored within the compute object
        const auto computed = [&]() {
          PointType point;
          specfem::compute::load_on_host(index, properties, point);
          return point;
        }();

        if (computed != expected) {
          std::ostringstream message;

          message << "\n \t Error in function check_compute_to_mesh";

          message << "\n \t Error at ispec = " << ispec << ", iz = " << index.iz
                  << ", ix = " << index.ix << "\n";
          message << "Expected: " << expected.print();
          message << "Got: " << computed.print();
          throw std::runtime_error(message.str());
        }
      });
}

TEST_F(ASSEMBLY, properties_access_functions) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);

    try {
      type_real offset = 10.1; // Random offset to store in the properties
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            set_value<_medium_tag_, _property_tag_, false>(elements, assembly,
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
            check_value<_medium_tag_, _property_tag_, false>(elements, assembly,
                                                             offset);
          });

      // SIMD access functions

      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            set_value<_medium_tag_, _property_tag_, true>(elements, assembly,
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
            check_value<_medium_tag_, _property_tag_, true>(elements, assembly,
                                                            offset);
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

TEST_F(ASSEMBLY, properties_construction) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);

    try {
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            check_compute_to_mesh<_medium_tag_, _property_tag_>(assembly, mesh);
          })

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

TEST_F(ASSEMBLY, properties_io_routines) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);

    // get current working directory
    // Access environment variable BUILD_DIR
    std::string temp_io_directory =
        (std::getenv("BUILD_DIR")
             ? std::string(std::getenv("BUILD_DIR"))
             : boost::filesystem::current_path().string()) +
        +"/tests/unit-tests/" + "temp_properties_io";
    boost::filesystem::create_directories(temp_io_directory);

    try {
      // Set all properties to a random value
      const type_real random_value = 10.1;
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            set_value<_medium_tag_, _property_tag_, false>(elements, assembly,
                                                           random_value);
          });

      // Copy properties to device
      assembly.properties.copy_to_device();

      // Create a property writer
      specfem::io::property_writer<specfem::io::ASCII<specfem::io::write> >
          writer(temp_io_directory);

      writer.write(assembly);

      // Create a property reader
      specfem::io::property_reader<specfem::io::ASCII<specfem::io::read> >
          reader(temp_io_directory);
      reader.read(assembly);

      // Check that the properties are the same
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2),
           MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
           PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
          {
            const auto elements = assembly.element_types.get_elements_on_host(
                _medium_tag_, _property_tag_);
            check_value<_medium_tag_, _property_tag_, false>(elements, assembly,
                                                             random_value);
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

    // Clean up the test file
    boost::filesystem::remove_all(temp_io_directory);
  }
}
