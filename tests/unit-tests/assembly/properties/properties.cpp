#include "../test_fixture/test_fixture.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/execution.hpp"
#include "specfem/io.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include <algorithm>
#include <cmath>
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
    const specfem::mesh::cartesian2d_mesh &mesh) {

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

namespace specfem::assembly_test {

/**
 * @brief Plain host copies of the attenuation-relevant state of one
 *        attenuating (medium, property) group.
 *
 * Snapshots are taken before and after a property write/read cycle so the
 * round trip can be verified against construction-time state.
 */
struct AttenuationStateSnapshot {
  using ScalarView = Kokkos::View<type_real *, Kokkos::HostSpace>;
  using GllView =
      Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using SlsView =
      Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>;

  ScalarView kappa_scale; ///< Per-element bulk modulus scale factors
  ScalarView mu_scale;    ///< Per-element shear modulus scale factors
  GllView Qkappa;         ///< Per-GLL bulk quality factors
  GllView Qmu;            ///< Per-GLL shear quality factors
  GllView kappa;          ///< Unrelaxed bulk moduli from the property container
  GllView mu;             ///< Unrelaxed shear moduli from the property
                          ///< container
  SlsView kappa_relaxation_rate; ///< Per-GLL, per-SLS bulk relaxation rates
  SlsView mu_relaxation_rate;    ///< Per-GLL, per-SLS shear relaxation rates
  int offset = 0; ///< Group offset into the property container views
};

/**
 * @brief Copy the attenuation state of one (medium, property) group into
 *        plain host views.
 *
 * @tparam MediumTag Medium tag of the attenuating group
 * @tparam PropertyTag Property tag of the attenuating group
 * @param assembly Assembly holding the attenuation and property containers
 * @return Snapshot of scale factors, quality factors, moduli and relaxation
 *         rates
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
AttenuationStateSnapshot take_attenuation_snapshot(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly) {
  const auto &att =
      assembly.attenuation.get_container<MediumTag, PropertyTag>();
  const auto &props =
      assembly.properties.get_container<MediumTag, PropertyTag>();

  const int nspec_attn = att.element_range.size();
  const int ngllz = att.h_Qkappa.extent(1);
  const int ngllx = att.h_Qkappa.extent(2);
  const int n_sls = att.h_kappa_relaxation_rate.extent(3);

  AttenuationStateSnapshot snapshot;
  snapshot.kappa_scale =
      AttenuationStateSnapshot::ScalarView("kappa_scale", nspec_attn);
  snapshot.mu_scale =
      AttenuationStateSnapshot::ScalarView("mu_scale", nspec_attn);
  snapshot.Qkappa =
      AttenuationStateSnapshot::GllView("Qkappa", nspec_attn, ngllz, ngllx);
  snapshot.Qmu =
      AttenuationStateSnapshot::GllView("Qmu", nspec_attn, ngllz, ngllx);
  snapshot.kappa =
      AttenuationStateSnapshot::GllView("kappa", nspec_attn, ngllz, ngllx);
  snapshot.mu =
      AttenuationStateSnapshot::GllView("mu", nspec_attn, ngllz, ngllx);
  snapshot.kappa_relaxation_rate = AttenuationStateSnapshot::SlsView(
      "kappa_relaxation_rate", nspec_attn, ngllz, ngllx, n_sls);
  snapshot.mu_relaxation_rate = AttenuationStateSnapshot::SlsView(
      "mu_relaxation_rate", nspec_attn, ngllz, ngllx, n_sls);
  snapshot.offset =
      att.element_range.begin_index() - props.element_range.begin_index();

  for (int i = 0; i < nspec_attn; ++i) {
    snapshot.kappa_scale(i) = att.h_kappa_scale(i);
    snapshot.mu_scale(i) = att.h_mu_scale(i);
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int ix = 0; ix < ngllx; ++ix) {
        snapshot.Qkappa(i, iz, ix) = att.h_Qkappa(i, iz, ix);
        snapshot.Qmu(i, iz, ix) = att.h_Qmu(i, iz, ix);
        snapshot.kappa(i, iz, ix) = props.h_kappa(snapshot.offset + i, iz, ix);
        snapshot.mu(i, iz, ix) = props.h_mu(snapshot.offset + i, iz, ix);
        for (int j = 0; j < n_sls; ++j) {
          snapshot.kappa_relaxation_rate(i, iz, ix, j) =
              att.h_kappa_relaxation_rate(i, iz, ix, j);
          snapshot.mu_relaxation_rate(i, iz, ix, j) =
              att.h_mu_relaxation_rate(i, iz, ix, j);
        }
      }
    }
  }

  return snapshot;
}

/**
 * @brief Compare two plain contiguous host views element-wise with a relative
 *        tolerance.
 *
 * The tolerance for each element is @c rel_tol scaled by the larger magnitude
 * of the two values (floored at 1 so near-zero values compare absolutely).
 *
 * @tparam ViewTypeA Plain LayoutRight host view type
 * @tparam ViewTypeB Plain LayoutRight host view type of equal size
 * @param computed View holding the values under test
 * @param expected View holding the reference values
 * @param rel_tol Relative tolerance
 * @param label Context prepended to failure messages
 */
template <typename ViewTypeA, typename ViewTypeB>
void expect_views_near(const ViewTypeA &computed, const ViewTypeB &expected,
                       const type_real rel_tol, const std::string &label) {
  ASSERT_EQ(computed.size(), expected.size()) << label << ": size mismatch";
  const auto *computed_data = computed.data();
  const auto *expected_data = expected.data();
  for (std::size_t i = 0; i < computed.size(); ++i) {
    const type_real magnitude =
        std::max({ std::abs(computed_data[i]), std::abs(expected_data[i]),
                   static_cast<type_real>(1.0) });
    EXPECT_NEAR(computed_data[i], expected_data[i], rel_tol * magnitude)
        << label << ": flat index " << i;
  }
}

} // namespace specfem::assembly_test

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
      // Set all properties to a random value
      const type_real random_value = 10.1;
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
                elements, assembly, random_value);
          });

      // Copy properties to device
      assembly.properties.copy_to_device();

      // Create a property writer
      specfem::io::property_writer<
          specfem::io_backends::ASCII<specfem::io::write>>
          writer(temp_io_directory);

      writer.write(assembly);

      // The writer stages attenuating datasets through scratch copies
      // (to_physical); the in-memory properties must be untouched.
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

TEST_F(Assembly2D, AttenuationPropertiesIORoundTrip) {
  // Construction-time state is mutually consistent (properties, quality
  // factors, scale factors and relaxation rates all derive from the same
  // material model), so the full attenuation state is a fixed point of the
  // write -> read cycle and can be compared against a pre-write snapshot.
  //
  // Tolerances: the ASCII backend stores 10 significant digits (exact for
  // float type_real, truncating for double), hence the tolerant comparisons.
  // Scale factors and relaxation rates get a looser tolerance because
  // recompute() re-solves tau_epsilon (Nelder-Mead) from the rounded
  // quality factors.
  constexpr type_real exact_tol = 1e-12;
  constexpr type_real roundtrip_tol = 1e-6;
  constexpr type_real recompute_tol = 1e-5;

  using AttenuationType =
      specfem::assembly::Attenuation<specfem::element::dimension_tag::dim2>;

  int n_groups_checked = 0;

  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);

    if (!Test.is_attenuation_enabled()) {
      continue;
    }

    // Snapshot every attenuating group before the write. Groups without
    // elements are skipped here and in the index-aligned loops below.
    std::vector<specfem::assembly_test::AttenuationStateSnapshot> before;
    std::vector<std::string> group_names;
    specfem::tag_dispatch::for_each(
        AttenuationType::attenuation_medium_combinations,
        [&]<typename TagsType>() {
          const auto &att =
              assembly.attenuation.get_container<TagsType::medium_tag,
                                                 TagsType::property_tag>();
          if (att.element_range.size() == 0) {
            return;
          }
          before.push_back(
              specfem::assembly_test::take_attenuation_snapshot<
                  TagsType::medium_tag, TagsType::property_tag>(assembly));
          group_names.push_back(specfem::element::to_string(
              TagsType::medium_tag, TagsType::property_tag,
              TagsType::attenuation_tag));
        });

    if (before.empty()) {
      // Nothing attenuates in this mesh; the generic round trip is covered
      // by properties_io_routines.
      continue;
    }

    // Distinct temp directory per config so parallel ctest invocations of
    // this binary cannot collide with properties_io_routines or each other.
    const std::string temp_io_directory =
        (std::getenv("BUILD_DIR")
             ? std::string(std::getenv("BUILD_DIR"))
             : boost::filesystem::current_path().string()) +
        "/tests/unit-tests/temp_properties_attenuation_io_" + suffix;
    boost::filesystem::create_directories(temp_io_directory);

    try {
      specfem::io::property_writer<
          specfem::io_backends::ASCII<specfem::io::write>>
          writer(temp_io_directory);
      writer.write(assembly);

      // The writer stages attenuating datasets through scratch copies
      // (to_physical); the in-memory state must be untouched.
      std::size_t igroup = 0;
      specfem::tag_dispatch::for_each(
          AttenuationType::attenuation_medium_combinations,
          [&]<typename TagsType>() {
            const auto &att =
                assembly.attenuation.get_container<TagsType::medium_tag,
                                                   TagsType::property_tag>();
            if (att.element_range.size() == 0) {
              return;
            }
            const auto mid = specfem::assembly_test::take_attenuation_snapshot<
                TagsType::medium_tag, TagsType::property_tag>(assembly);
            const std::string label =
                Test.name + " / " + group_names[igroup] + " / post-write ";
            specfem::assembly_test::expect_views_near(
                mid.kappa, before[igroup].kappa, exact_tol, label + "kappa");
            specfem::assembly_test::expect_views_near(mid.mu, before[igroup].mu,
                                                      exact_tol, label + "mu");
            specfem::assembly_test::expect_views_near(
                mid.kappa_scale, before[igroup].kappa_scale, exact_tol,
                label + "kappa_scale");
            specfem::assembly_test::expect_views_near(
                mid.mu_scale, before[igroup].mu_scale, exact_tol,
                label + "mu_scale");
            specfem::assembly_test::expect_views_near(
                mid.Qkappa, before[igroup].Qkappa, exact_tol, label + "Qkappa");
            specfem::assembly_test::expect_views_near(
                mid.Qmu, before[igroup].Qmu, exact_tol, label + "Qmu");
            ++igroup;
          });

      // The file must hold the physical (relaxed) moduli — the staged
      // unrelaxed values divided by the per-element scale factors — and
      // the quality factors verbatim. Reading the datasets back directly
      // isolates the writer from the reader.
      igroup = 0;
      specfem::tag_dispatch::for_each(
          AttenuationType::attenuation_medium_combinations,
          [&]<typename TagsType>() {
            const auto &att =
                assembly.attenuation.get_container<TagsType::medium_tag,
                                                   TagsType::property_tag>();
            if (att.element_range.size() == 0) {
              return;
            }
            const auto &snapshot = before[igroup];
            const int nspec_attn = snapshot.Qkappa.extent(0);
            const int ngllz = snapshot.Qkappa.extent(1);
            const int ngllx = snapshot.Qkappa.extent(2);

            using GllView =
                specfem::assembly_test::AttenuationStateSnapshot::GllView;
            GllView disk_kappa("disk_kappa", nspec_attn, ngllz, ngllx);
            GllView disk_mu("disk_mu", nspec_attn, ngllz, ngllx);
            GllView disk_Qkappa("disk_Qkappa", nspec_attn, ngllz, ngllx);
            GllView disk_Qmu("disk_Qmu", nspec_attn, ngllz, ngllx);

            typename specfem::io_backends::ASCII<specfem::io::read>::File file(
                temp_io_directory + "/Properties");
            auto group = file.openGroup("/" + group_names[igroup]);
            group.openDataset("kappa", disk_kappa).read();
            group.openDataset("mu", disk_mu).read();
            group.openDataset("Qkappa", disk_Qkappa).read();
            group.openDataset("Qmu", disk_Qmu).read();

            GllView expected_kappa("expected_kappa", nspec_attn, ngllz, ngllx);
            GllView expected_mu("expected_mu", nspec_attn, ngllz, ngllx);
            for (int i = 0; i < nspec_attn; ++i) {
              for (int iz = 0; iz < ngllz; ++iz) {
                for (int ix = 0; ix < ngllx; ++ix) {
                  expected_kappa(i, iz, ix) =
                      snapshot.kappa(i, iz, ix) / snapshot.kappa_scale(i);
                  expected_mu(i, iz, ix) =
                      snapshot.mu(i, iz, ix) / snapshot.mu_scale(i);
                }
              }
            }

            const std::string label =
                Test.name + " / " + group_names[igroup] + " / on-disk ";
            specfem::assembly_test::expect_views_near(
                disk_kappa, expected_kappa, roundtrip_tol, label + "kappa");
            specfem::assembly_test::expect_views_near(
                disk_mu, expected_mu, roundtrip_tol, label + "mu");
            specfem::assembly_test::expect_views_near(
                disk_Qkappa, snapshot.Qkappa, roundtrip_tol, label + "Qkappa");
            specfem::assembly_test::expect_views_near(
                disk_Qmu, snapshot.Qmu, roundtrip_tol, label + "Qmu");
            ++igroup;
          });

      // Read back: restores physical moduli from the file, then recompute()
      // rescales them to unrelaxed values and rebuilds relaxation rates and
      // scale factors from the read quality factors.
      specfem::io::property_reader<
          specfem::io_backends::ASCII<specfem::io::read>>
          reader(temp_io_directory);
      reader.read(assembly);

      igroup = 0;
      specfem::tag_dispatch::for_each(
          AttenuationType::attenuation_medium_combinations,
          [&]<typename TagsType>() {
            const auto &att =
                assembly.attenuation.get_container<TagsType::medium_tag,
                                                   TagsType::property_tag>();
            if (att.element_range.size() == 0) {
              return;
            }
            const auto after =
                specfem::assembly_test::take_attenuation_snapshot<
                    TagsType::medium_tag, TagsType::property_tag>(assembly);
            const std::string label =
                Test.name + " / " + group_names[igroup] + " / post-read ";
            specfem::assembly_test::expect_views_near(
                after.kappa, before[igroup].kappa, roundtrip_tol,
                label + "kappa");
            specfem::assembly_test::expect_views_near(
                after.mu, before[igroup].mu, roundtrip_tol, label + "mu");
            specfem::assembly_test::expect_views_near(
                after.Qkappa, before[igroup].Qkappa, roundtrip_tol,
                label + "Qkappa");
            specfem::assembly_test::expect_views_near(
                after.Qmu, before[igroup].Qmu, roundtrip_tol, label + "Qmu");
            specfem::assembly_test::expect_views_near(
                after.kappa_scale, before[igroup].kappa_scale, recompute_tol,
                label + "kappa_scale");
            specfem::assembly_test::expect_views_near(
                after.mu_scale, before[igroup].mu_scale, recompute_tol,
                label + "mu_scale");
            specfem::assembly_test::expect_views_near(
                after.kappa_relaxation_rate,
                before[igroup].kappa_relaxation_rate, recompute_tol,
                label + "kappa_relaxation_rate");
            specfem::assembly_test::expect_views_near(
                after.mu_relaxation_rate, before[igroup].mu_relaxation_rate,
                recompute_tol, label + "mu_relaxation_rate");
            ++igroup;
          });

      n_groups_checked += static_cast<int>(before.size());

      if (::testing::Test::HasFailure()) {
        std::cout
            << "-------------------------------------------------------\n"
            << "\033[0;31m[FAILED]\033[0m " << Test.name << "\n"
            << "-------------------------------------------------------\n\n"
            << std::endl;
      } else {
        std::cout
            << "-------------------------------------------------------\n"
            << "\033[0;32m[PASSED]\033[0m " << Test.name << "\n"
            << "-------------------------------------------------------\n\n"
            << std::endl;
      }
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

    // Clean up the test files
    boost::filesystem::remove_all(temp_io_directory);
  }

  EXPECT_GT(n_groups_checked, 0)
      << "no attenuating group was exercised by any test configuration";
}
