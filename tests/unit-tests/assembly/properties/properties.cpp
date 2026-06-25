#include "../test_fixture/test_fixture.hpp"
#include "specfem/constants.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/execution.hpp"
#include "specfem/io.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

template <bool using_simd, typename ExecutionSpace>
using ParallelConfig = specfem::parallel_configuration::default_chunk_config<
    specfem::element::dimension_tag::dim2,
    specfem::datatype::simd<type_real, using_simd>, ExecutionSpace>;

// Compile-time checks of the attenuating-combination predicate used to guard
// Q I/O in the property writer/reader.
static_assert(
    specfem::assembly::Attenuation<specfem::element::dimension_tag::dim2>::
        has_attenuation<specfem::element::medium_tag::elastic_psv,
                        specfem::element::property_tag::isotropic>(),
    "elastic_psv/isotropic must be an attenuating combination in 2D");
static_assert(
    !specfem::assembly::Attenuation<specfem::element::dimension_tag::dim2>::
        has_attenuation<specfem::element::medium_tag::acoustic,
                        specfem::element::property_tag::isotropic>(),
    "acoustic must not be an attenuating combination in 2D");
static_assert(
    specfem::assembly::Attenuation<specfem::element::dimension_tag::dim3>::
        has_attenuation<specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic>(),
    "elastic/isotropic must be an attenuating combination in 3D");
static_assert(
    !specfem::assembly::Attenuation<specfem::element::dimension_tag::dim3>::
        has_attenuation<specfem::element::medium_tag::acoustic,
                        specfem::element::property_tag::isotropic>(),
    "acoustic must not be an attenuating combination in 3D");

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ExecutionSpace, typename ViewType>
std::enable_if_t<
    std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>, void>
set_property_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const type_real offset) {

  const auto &properties = assembly.properties;

  using PointPropertiesType = specfem::point::properties<
      specfem::tags::Tags<specfem::element::dimension_tag::dim2, MediumTag,
                          PropertyTag, using_simd>>;
  using PointType = typename PointPropertiesType::value_type;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  specfem::execution::for_all(
      "set_to_value", policy,
      [=](const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        PointPropertiesType point(static_cast<type_real>(index.ispec + offset));
        specfem::assembly::store_on_host(index, point, properties);
      });

  Kokkos::fence();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ExecutionSpace, typename ViewType>
std::enable_if_t<
    std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>, void>
check_property_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const type_real offset) {

  const auto &properties = assembly.properties;
  using PointType = specfem::point::properties<
      specfem::tags::Tags<specfem::element::dimension_tag::dim2, MediumTag,
                          PropertyTag, using_simd>>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultHostExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  specfem::execution::for_all(
      "set_to_value", policy,
      [=](const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        using datatype = typename PointType::value_type;

        PointType expected;

        if constexpr (using_simd) {
          datatype value([&](const std::size_t lane) {
            return (lane < index.number_elements)
                       ? static_cast<type_real>(index.ispec + offset)
                       : static_cast<type_real>(0.0);
          });
          expected = value;
        } else {
          datatype value = static_cast<type_real>(index.ispec + offset);
          expected = value;
        }

        PointType point_poperties_computed;
        specfem::assembly::load_on_host(index, properties,
                                        point_poperties_computed);

        if (point_poperties_computed != expected) {
          std::ostringstream message;

          message << "\n \t Error in function check_to_value";

          message << "\n \t Error at ispec = " << index.ispec
                  << ", iz = " << index.iz << ", ix = " << index.ix << "\n";
          message << "Expected: " << expected.print();
          message << "Got: " << point_poperties_computed.print();
          throw std::runtime_error(message.str());
        }
      });

  Kokkos::fence();
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename ExecutionSpace, typename ViewType>
std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>,
                 void>
check_property_value(
    const ViewType elements,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const type_real offset) {

  const int nspec = assembly.mesh.nspec;
  const int ngll = assembly.mesh.element_grid.ngllx;
  const auto &properties = assembly.properties;

  using PointType = specfem::point::properties<
      specfem::tags::Tags<specfem::element::dimension_tag::dim2, MediumTag,
                          PropertyTag, using_simd>>;

  Kokkos::View<PointType ***, Kokkos::DefaultExecutionSpace> point_view(
      "point_view", nspec, ngll, ngll);

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<using_simd, Kokkos::DefaultExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  specfem::execution::for_all(
      "set_to_value", policy,
      KOKKOS_LAMBDA(
          const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        PointType computed;
        specfem::assembly::load_on_device(index, properties, computed);

        const int ispec = index.ispec;
        const int iz = index.iz;
        const int ix = index.ix;

        point_view(ispec, iz, ix) = computed;
      });

  Kokkos::fence();
  const auto point_view_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultHostExecutionSpace(), point_view);

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

  specfem::execution::for_all(
      "set_to_value", host_policy,
      [=](const typename decltype(host_policy)::base_index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        PointType expected(static_cast<type_real>(index.ispec + offset));
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
#endif

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::attenuation_tag AttenuationTag>
void check_compute_to_mesh(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    const specfem::mesh::mesh<specfem::element::dimension_tag::dim2> &mesh) {

  const auto &properties = assembly.properties;
  const auto &element_types = assembly.element_types;
  const auto &mesh_assembly = assembly.mesh;
  const auto &materials = mesh.materials;

  // Get all elements of the given type
  const auto elements = element_types.get_elements_on_host(
      MediumTag, PropertyTag, AttenuationTag);

  using PointType = specfem::point::properties<specfem::tags::Tags<
      specfem::element::dimension_tag::dim2, MediumTag, PropertyTag, false>>;

  specfem::execution::ChunkedDomainIterator policy(
      ParallelConfig<false, Kokkos::DefaultHostExecutionSpace>(), elements,
      assembly.mesh.element_grid);

  specfem::execution::for_all(
      "set_to_value", policy,
      [=](const typename decltype(policy)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ispec = index.ispec;

        // Get the properties stored within the mesh
        const int ispec_mesh = mesh_assembly.h_compute_to_mesh(ispec);
        const auto expected =
            materials
                .get_material<MediumTag, PropertyTag, AttenuationTag>(
                    ispec_mesh)
                .get_properties();

        // Get the properties stored within the compute object
        const auto computed = [&]() {
          PointType point;
          specfem::assembly::load_on_host(index, properties, point);
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

TEST_F(Assembly2D, properties_access_functions) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);

    try {
      type_real offset = 10.1; // Random offset to store in the properties
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                         elastic_psv_t) *
              PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag);
            set_property_value<ElementTags::medium_tag,
                               ElementTags::property_tag, false,
                               Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, offset);
          });

      // Check that we are able to access the values stored in the properties
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                         elastic_psv_t) *
              PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag);
            check_property_value<ElementTags::medium_tag,
                                 ElementTags::property_tag, false,
                                 Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, offset);
          });

      // SIMD access functions

      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                         elastic_psv_t) *
              PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag);
            set_property_value<ElementTags::medium_tag,
                               ElementTags::property_tag, true,
                               Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, offset);
          });

      // Check that we are able to access the values stored in the properties
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                         elastic_psv_t) *
              PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag);
            check_property_value<ElementTags::medium_tag,
                                 ElementTags::property_tag, true,
                                 Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, offset);
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

TEST_F(Assembly2D, properties_construction) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);

    try {
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                         elastic_psv_t) *
              PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
              ATTENUATION_SET(none, constant_isotropic),
          [&]<typename ElementTags>() {
            check_compute_to_mesh<ElementTags::medium_tag,
                                  ElementTags::property_tag,
                                  ElementTags::attenuation_tag>(assembly, mesh);
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

TEST_F(Assembly2D, properties_io_routines) {
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
      // Helper: iterate the per-GLL attenuation quality factors and modulus
      // scale factors for any medium that attenuates and has attenuating
      // elements. No-op for non-attenuating configs (extent == 0). The callback
      // receives mutable refs (Qkappa, Qmu, kappa_scale, mu_scale).
      const auto for_each_attenuation_q = [&](auto fn) {
        specfem::tag_dispatch::for_each(
            DIMENSION_SET(dim2) *
                MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                           elastic_psv_t) *
                PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
            [&]<typename ElementTags>() {
              constexpr auto medium_tag = ElementTags::medium_tag;
              constexpr auto property_tag = ElementTags::property_tag;
              if constexpr (decltype(assembly.attenuation)::
                                template has_attenuation<medium_tag,
                                                         property_tag>()) {
                const auto &att =
                    assembly.attenuation
                        .get_container<medium_tag, property_tag>();
                if (att.h_Qkappa.extent(0) == 0)
                  return;
                for (std::size_t i = 0; i < att.h_Qkappa.extent(0); ++i)
                  for (std::size_t iz = 0; iz < att.h_Qkappa.extent(1); ++iz)
                    for (std::size_t ix = 0; ix < att.h_Qkappa.extent(2); ++ix)
                      fn(att.h_Qkappa(i, iz, ix), att.h_Qmu(i, iz, ix),
                         att.h_kappa_scale(i, iz, ix),
                         att.h_mu_scale(i, iz, ix));
              }
            });
      };

      // Helper: iterate the per-(GLL, SLS) relaxation rates for any attenuating
      // medium. The callback receives mutable refs (kappa_rate, mu_rate).
      const auto for_each_attenuation_rate = [&](auto fn) {
        specfem::tag_dispatch::for_each(
            DIMENSION_SET(dim2) *
                MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                           elastic_psv_t) *
                PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
            [&]<typename ElementTags>() {
              constexpr auto medium_tag = ElementTags::medium_tag;
              constexpr auto property_tag = ElementTags::property_tag;
              if constexpr (decltype(assembly.attenuation)::
                                template has_attenuation<medium_tag,
                                                         property_tag>()) {
                const auto &att =
                    assembly.attenuation
                        .get_container<medium_tag, property_tag>();
                const auto &kr = att.h_kappa_relaxation_rate;
                const auto &mr = att.h_mu_relaxation_rate;
                if (kr.extent(0) == 0)
                  return;
                for (std::size_t i = 0; i < kr.extent(0); ++i)
                  for (std::size_t iz = 0; iz < kr.extent(1); ++iz)
                    for (std::size_t ix = 0; ix < kr.extent(2); ++ix)
                      for (std::size_t j = 0; j < kr.extent(3); ++j)
                        fn(kr(i, iz, ix, j), mr(i, iz, ix, j));
              }
            });
      };

      // Re-set every property buffer (kappa/mu/rho) to a known value, so the
      // post-read check is meaningful (a no-op reader would otherwise pass).
      const auto set_all_properties = [&](const type_real offset) {
        specfem::tag_dispatch::for_each(
            DIMENSION_SET(dim2) *
                MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                           elastic_psv_t) *
                PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
            [&]<typename ElementTags>() {
              const auto elements = assembly.element_types.get_elements_on_host(
                  ElementTags::medium_tag, ElementTags::property_tag);
              set_property_value<ElementTags::medium_tag,
                                 ElementTags::property_tag, false,
                                 Kokkos::DefaultHostExecutionSpace>(
                  elements, assembly, offset);
            });
      };

      // Set all properties to a known value (per point: ispec + random_value).
      const type_real random_value = 10.1;
      set_all_properties(random_value);

      // Snapshot the attenuation quality factors and scale factors before
      // writing, and confirm the scale factors are non-trivial (so the
      // write/read (un)scaling is actually exercised by attenuating configs).
      std::vector<type_real> saved_qkappa, saved_qmu, saved_kappa_scale,
          saved_mu_scale;
      bool any_nontrivial_scale = false;
      for_each_attenuation_q([&](type_real &qkappa, type_real &qmu,
                                 type_real &kappa_scale, type_real &mu_scale) {
        saved_qkappa.push_back(qkappa);
        saved_qmu.push_back(qmu);
        saved_kappa_scale.push_back(kappa_scale);
        saved_mu_scale.push_back(mu_scale);
        if (std::abs(kappa_scale - 1) > 1e-6 || std::abs(mu_scale - 1) > 1e-6)
          any_nontrivial_scale = true;
      });
      if (!saved_kappa_scale.empty())
        EXPECT_TRUE(any_nontrivial_scale)
            << "attenuating config should have non-unit modulus scale factors";

      // Snapshot the construction-time relaxation rates so we can confirm the
      // reader recomputes them from the read-back Q (see check below).
      std::vector<type_real> saved_kappa_rate, saved_mu_rate;
      for_each_attenuation_rate([&](type_real &kappa_rate, type_real &mu_rate) {
        saved_kappa_rate.push_back(kappa_rate);
        saved_mu_rate.push_back(mu_rate);
      });

      // Copy properties to device
      assembly.properties.copy_to_device();

      // Create a property writer
      specfem::io::property_writer<
          specfem::io_backends::ASCII<specfem::io::write>>
          writer(temp_io_directory);

      writer.write(assembly);

      // Corrupt the host Q views and the scale factors so a successful read
      // (which re-reads Q and recomputes the scale) is unambiguous.
      for_each_attenuation_q([&](type_real &qkappa, type_real &qmu,
                                 type_real &kappa_scale, type_real &mu_scale) {
        qkappa = static_cast<type_real>(-999);
        qmu = static_cast<type_real>(-999);
        kappa_scale = static_cast<type_real>(-999);
        mu_scale = static_cast<type_real>(-999);
      });

      // Corrupt the relaxation rates too, so the post-read check confirms the
      // reader recomputed them (rather than leaving the corrupt values).
      for_each_attenuation_rate([&](type_real &kappa_rate, type_real &mu_rate) {
        kappa_rate = static_cast<type_real>(-999);
        mu_rate = static_cast<type_real>(-999);
      });

      // Corrupt every property buffer too, so a no-op reader would fail the
      // round-trip check below (the buffer no longer holds random_value).
      set_all_properties(static_cast<type_real>(-98765));
      assembly.properties.copy_to_device();

      // Create a property reader
      specfem::io::property_reader<
          specfem::io_backends::ASCII<specfem::io::read>>
          reader(temp_io_directory);
      reader.read(assembly);

      // Check that the properties are the same
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                         elastic_psv_t) *
              PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
          [&]<typename ElementTags>() {
            const auto elements = assembly.element_types.get_elements_on_host(
                ElementTags::medium_tag, ElementTags::property_tag);
            check_property_value<ElementTags::medium_tag,
                                 ElementTags::property_tag, false,
                                 Kokkos::DefaultHostExecutionSpace>(
                elements, assembly, random_value);
          });

      // Verify the attenuation quality factors round-tripped and that the
      // reader recomputed the same scale factors from the read-back Q. No-op
      // (and empty snapshot) for non-attenuating configs.
      std::size_t q_index = 0;
      for_each_attenuation_q([&](type_real &qkappa, type_real &qmu,
                                 type_real &kappa_scale, type_real &mu_scale) {
        EXPECT_NEAR(qkappa, saved_qkappa[q_index], 1e-4);
        EXPECT_NEAR(qmu, saved_qmu[q_index], 1e-4);
        EXPECT_NEAR(kappa_scale, saved_kappa_scale[q_index], 1e-4);
        EXPECT_NEAR(mu_scale, saved_mu_scale[q_index], 1e-4);
        ++q_index;
      });

      // Verify the reader recomputed the relaxation rates from the read-back Q.
      // The relaxation rate is modulus * factor(Q); the moduli were
      // overwritten, so absolute values differ, but the per-SLS factor ratio
      // (Q-dependent, modulus-independent) must be preserved -- and the rates
      // must no longer be the corrupt sentinel.
      std::vector<type_real> read_kappa_rate, read_mu_rate;
      for_each_attenuation_rate([&](type_real &kappa_rate, type_real &mu_rate) {
        read_kappa_rate.push_back(kappa_rate);
        read_mu_rate.push_back(mu_rate);
      });
      ASSERT_EQ(read_kappa_rate.size(), saved_kappa_rate.size());
      constexpr std::size_t n_sls = specfem::constants::N_SLS;
      for (std::size_t g = 0; g + n_sls <= read_kappa_rate.size(); g += n_sls) {
        for (std::size_t j = 0; j < n_sls; ++j) {
          EXPECT_NE(read_kappa_rate[g + j], static_cast<type_real>(-999));
          EXPECT_TRUE(std::isfinite(read_kappa_rate[g + j]));
          EXPECT_TRUE(std::isfinite(read_mu_rate[g + j]));
          // Per-SLS factor ratio (depends only on Q) preserved across read.
          if (saved_kappa_rate[g + j] != 0 && saved_kappa_rate[g] != 0)
            EXPECT_NEAR(read_kappa_rate[g + j] / saved_kappa_rate[g + j],
                        read_kappa_rate[g] / saved_kappa_rate[g],
                        std::abs(read_kappa_rate[g] / saved_kappa_rate[g]) *
                            1e-3);
          if (saved_mu_rate[g + j] != 0 && saved_mu_rate[g] != 0)
            EXPECT_NEAR(read_mu_rate[g + j] / saved_mu_rate[g + j],
                        read_mu_rate[g] / saved_mu_rate[g],
                        std::abs(read_mu_rate[g] / saved_mu_rate[g]) * 1e-3);
        }
      }

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
