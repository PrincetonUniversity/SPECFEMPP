#include "../test_fixture/test_fixture.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/execution.hpp"
#include "specfem/io.hpp"
#include "specfem/io/property/impl/sub_block.hpp"
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
static_assert(
    specfem::assembly::Attenuation<specfem::element::dimension_tag::dim2>::
        has_attenuation<
            specfem::element::medium_tag::elastic_psv,
            specfem::element::property_tag::isotropic,
            specfem::element::attenuation_tag::constant_isotropic>(),
    "elastic_psv/isotropic/constant_isotropic must have a container in 2D");
static_assert(
    !specfem::assembly::Attenuation<specfem::element::dimension_tag::dim2>::
        has_attenuation<specfem::element::medium_tag::elastic_psv,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>(),
    "elastic_psv/isotropic/none must not have a container in 2D");
static_assert(
    !specfem::assembly::Attenuation<specfem::element::dimension_tag::dim2>::
        has_attenuation<
            specfem::element::medium_tag::acoustic,
            specfem::element::property_tag::isotropic,
            specfem::element::attenuation_tag::constant_isotropic>(),
    "acoustic/isotropic/constant_isotropic must not have a container in 2D");
static_assert(
    specfem::assembly::Attenuation<specfem::element::dimension_tag::dim3>::
        has_attenuation<
            specfem::element::medium_tag::elastic,
            specfem::element::property_tag::isotropic,
            specfem::element::attenuation_tag::constant_isotropic>(),
    "elastic/isotropic/constant_isotropic must have a container in 3D");

// The property writer/reader serialize datasets in logical row-major order by
// staging chunk-tiled domain views through plain views (sub_block.hpp).
// Use an element count that does not divide the storage chunk size so the
// tail tile is padded: serializing the tiled storage directly would truncate
// real values interleaved into the padding -- a bug that only manifests on
// SIMD builds whose chunk width does not divide the element count.
TEST(PropertiesSubBlock, PackUnpackWithPaddedTile) {
  constexpr int chunk = specfem::parallel_configuration::storage_chunk_size;
  const int nspec = chunk + 3;
  const int ngllz = 5, ngllx = 5;

  using DomainView =
      specfem::datatype::DomainView<specfem::element::dimension_tag::dim2,
                                    type_real, 3, Kokkos::HostSpace>;
  DomainView view("sub_block_test", nspec, ngllz, ngllx);
  const auto value = [](const int e, const int iz, const int ix) {
    return static_cast<type_real>(e * 10000 + iz * 100 + ix);
  };
  for (int e = 0; e < nspec; ++e)
    for (int iz = 0; iz < ngllz; ++iz)
      for (int ix = 0; ix < ngllx; ++ix)
        view(e, iz, ix) = value(e, iz, ix);

  // Full-span pack: the plain view must hold the logical values in row-major
  // flat order (this is the payload the backends serialize).
  const auto packed =
      specfem::io::property_impl::extract_sub_block(view, "full", 0, nspec);
  for (int e = 0; e < nspec; ++e)
    for (int iz = 0; iz < ngllz; ++iz)
      for (int ix = 0; ix < ngllx; ++ix) {
        EXPECT_NEAR(packed(e, iz, ix), value(e, iz, ix), 1e-6);
        EXPECT_NEAR(packed.data()[(e * ngllz + iz) * ngllx + ix],
                    value(e, iz, ix), 1e-6);
      }

  // Partial pack at an offset (mixed-group sub-block).
  const int offset = 2;
  const int count = nspec - 3;
  const auto partial = specfem::io::property_impl::extract_sub_block(
      view, "partial", offset, count);
  for (int e = 0; e < count; ++e)
    for (int iz = 0; iz < ngllz; ++iz)
      for (int ix = 0; ix < ngllx; ++ix)
        EXPECT_NEAR(partial(e, iz, ix), value(offset + e, iz, ix), 1e-6);

  // Unpack restores the slice and leaves the rest untouched.
  DomainView restored("sub_block_restored", nspec, ngllz, ngllx);
  Kokkos::deep_copy(restored, static_cast<type_real>(-1));
  specfem::io::property_impl::insert_sub_block(restored, partial, offset);
  for (int e = 0; e < nspec; ++e)
    for (int iz = 0; iz < ngllz; ++iz)
      for (int ix = 0; ix < ngllx; ++ix) {
        const bool in_slice = (e >= offset && e < offset + count);
        EXPECT_NEAR(restored(e, iz, ix),
                    in_slice ? value(e, iz, ix) : static_cast<type_real>(-1),
                    1e-6);
      }
}

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

/**
 * @brief Visit every entry of every attenuation model dataset (reference
 *        moduli and Q) as fn(value_ref, gll_parity).
 *
 * gll_parity alternates between 0 and 1 across the GLL points of an element,
 * giving callers a deterministic per-GLL pattern. No-op for non-attenuating
 * configurations. Only the container's generic model-I/O interface is used,
 * so this is agnostic of the attenuation implementation.
 */
template <typename Fn>
void visit_attenuation_model(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    Fn fn) {
  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
      [&]<typename ElementTags>() {
        constexpr auto medium_tag = ElementTags::medium_tag;
        constexpr auto property_tag = ElementTags::property_tag;
        if constexpr (decltype(assembly.attenuation)::template has_attenuation<
                          medium_tag, property_tag>()) {
          const auto &att =
              assembly.attenuation.get_container<medium_tag, property_tag>();
          for (const auto &[view_name, view] : att.io_views()) {
            for (std::size_t e = 0; e < view.extent(0); ++e)
              for (std::size_t iz = 0; iz < view.extent(1); ++iz)
                for (std::size_t ix = 0; ix < view.extent(2); ++ix)
                  fn(view(e, iz, ix), static_cast<int>((iz + ix) % 2));
          }
        }
      });
}

/**
 * @brief Visit every entry of the runtime state that the reader derives from
 *        the on-disk model for attenuating groups: all property views of the
 *        (medium, property) group plus the recomputed relaxation-rate views.
 *
 * Used to verify read idempotence: the on-disk model fully determines this
 * state, so a second write/read round trip must reproduce it.
 */
template <typename Fn>
void visit_attenuating_group_state(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly,
    Fn fn) {
  const auto visit_entries = [&](const auto view) {
    using ViewType = std::decay_t<decltype(view)>;
    for (std::size_t e = 0; e < view.extent(0); ++e)
      for (std::size_t iz = 0; iz < view.extent(1); ++iz)
        for (std::size_t ix = 0; ix < view.extent(2); ++ix) {
          if constexpr (ViewType::rank() == 3) {
            fn(view(e, iz, ix));
          } else {
            for (std::size_t j = 0; j < view.extent(3); ++j)
              fn(view(e, iz, ix, j));
          }
        }
  };
  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
      [&]<typename ElementTags>() {
        constexpr auto medium_tag = ElementTags::medium_tag;
        constexpr auto property_tag = ElementTags::property_tag;
        if constexpr (decltype(assembly.attenuation)::template has_attenuation<
                          medium_tag, property_tag>()) {
          const auto &att =
              assembly.attenuation.get_container<medium_tag, property_tag>();
          if (att.element_range.size() == 0)
            return;
          const auto &container =
              assembly.properties.get_container<medium_tag, property_tag>();
          container.for_each_host_view(
              [&](const auto view, const std::string) { visit_entries(view); });
          att.for_each_recomputed_host_view(
              [&](const auto view, const std::string) { visit_entries(view); });
        }
      });
}

/**
 * @brief EXPECT that the recomputed attenuation state (e.g. modulus scale
 *        factors and relaxation rates) varies across the GLL points of every
 *        element.
 *
 * Used after reading back a GLL-varying model: the moduli are element-constant
 * in the test, so any GLL variation must come from the per-GLL model --
 * element-constant values would mean the reader collapsed the model (e.g.
 * broadcast each element's first GLL point).
 */
inline void expect_recomputed_state_varies(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly) {
  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
      [&]<typename ElementTags>() {
        constexpr auto medium_tag = ElementTags::medium_tag;
        constexpr auto property_tag = ElementTags::property_tag;
        if constexpr (decltype(assembly.attenuation)::template has_attenuation<
                          medium_tag, property_tag>()) {
          const auto &att =
              assembly.attenuation.get_container<medium_tag, property_tag>();
          att.for_each_recomputed_host_view(
              [&](const auto view, const std::string view_name) {
                for (std::size_t e = 0; e < view.extent(0); ++e) {
                  // Compare every GLL point of element e against its first
                  // one; rank-4 views carry a trailing per-point (SLS) index,
                  // sampled at 0 so only GLL variation counts.
                  bool varies = false;
                  for (std::size_t iz = 0; iz < view.extent(1); ++iz)
                    for (std::size_t ix = 0; ix < view.extent(2); ++ix) {
                      using ViewType = std::decay_t<decltype(view)>;
                      const type_real first = [&]() {
                        if constexpr (ViewType::rank() == 3)
                          return view(e, 0, 0);
                        else
                          return view(e, 0, 0, 0);
                      }();
                      const type_real value = [&]() {
                        if constexpr (ViewType::rank() == 3)
                          return view(e, iz, ix);
                        else
                          return view(e, iz, ix, 0);
                      }();
                      if (std::abs(value - first) > std::abs(first) * 1e-6)
                        varies = true;
                    }
                  EXPECT_TRUE(varies)
                      << view_name << " should vary across GLL points of "
                      << "element " << e << " for a GLL-varying model";
                }
              });
        }
      });
}

TEST_F(Assembly2D, properties_io_routines) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
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

      // Snapshot the attenuation model before writing; its datasets
      // (reference moduli, Q) are persisted verbatim so they must round-trip
      // through the file.
      std::vector<type_real> model_snapshot;
      visit_attenuation_model(assembly, [&](type_real &value, const int) {
        model_snapshot.push_back(value);
      });

      writer.write(assembly);

      // Create a property reader
      specfem::io::property_reader<
          specfem::io_backends::ASCII<specfem::io::read>>
          reader(temp_io_directory);
      reader.read(assembly);

      // Full property round trip holds for combinations whose datasets all
      // come from the property container. For attenuating combinations the
      // "kappa"/"mu" datasets are owned by the attenuation container (the
      // reference moduli) and the runtime moduli are recomputed from the
      // on-disk model on read, so those are validated separately below.
      specfem::tag_dispatch::for_each(
          DIMENSION_SET(dim2) *
              MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                         elastic_psv_t) *
              PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
              ATTENUATION_SET(none, constant_isotropic),
          [&]<typename ElementTags>() {
            constexpr auto medium_tag = ElementTags::medium_tag;
            constexpr auto property_tag = ElementTags::property_tag;
            constexpr auto attenuation_tag = ElementTags::attenuation_tag;
            if constexpr (!decltype(assembly.attenuation)::
                              template has_attenuation<medium_tag, property_tag,
                                                       attenuation_tag>()) {
              const auto elements = assembly.element_types.get_elements_on_host(
                  medium_tag, property_tag, attenuation_tag);
              if (elements.size() == 0)
                return;
              check_property_value<medium_tag, property_tag, false,
                                   Kokkos::DefaultHostExecutionSpace>(
                  elements, assembly, random_value);
            }
          });

      // The model datasets round-trip verbatim (up to ASCII quantization).
      std::size_t model_index = 0;
      visit_attenuation_model(assembly, [&](type_real &value, const int) {
        EXPECT_NEAR(value, model_snapshot[model_index],
                    std::abs(model_snapshot[model_index]) * 1e-6 + 1e-4);
        ++model_index;
      });

      // Read idempotence for attenuating groups: the on-disk model fully
      // determines the runtime moduli and relaxation rates, so a second
      // write/read round trip must reproduce them.
      std::vector<type_real> state_snapshot;
      visit_attenuating_group_state(assembly, [&](const type_real value) {
        state_snapshot.push_back(value);
      });
      if (!state_snapshot.empty()) {
        writer.write(assembly);
        reader.read(assembly);
        std::size_t state_index = 0;
        visit_attenuating_group_state(assembly, [&](const type_real value) {
          EXPECT_NEAR(value, state_snapshot[state_index],
                      std::abs(state_snapshot[state_index]) * 1e-6 + 1e-20);
          ++state_index;
        });
      }

      // Attenuation model datasets are stored per GLL point so a written
      // model can be edited GLL-by-GLL. Perturb the model per GLL, round-trip,
      // and verify the reader honours the per-GLL values instead of collapsing
      // them (e.g. broadcasting each element's first GLL point).
      if (Test.is_attenuation_enabled()) {
        std::vector<type_real> model;
        visit_attenuation_model(assembly, [&](type_real &value,
                                              const int gll_parity) {
          value *=
              static_cast<type_real>(1) +
              static_cast<type_real>(0.1) * static_cast<type_real>(gll_parity);
          model.push_back(value);
        });

        writer.write(assembly);
        reader.read(assembly);

        std::size_t index = 0;
        visit_attenuation_model(assembly, [&](type_real &value, const int) {
          EXPECT_NEAR(value, model[index],
                      std::abs(model[index]) * 1e-6 + 1e-4);
          ++index;
        });
        expect_recomputed_state_varies(assembly);
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
