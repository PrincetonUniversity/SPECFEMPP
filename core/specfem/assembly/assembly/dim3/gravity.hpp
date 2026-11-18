#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/globe/model_evaluator.hpp"

#include <Kokkos_Core.hpp>

namespace specfem::assembly {

template <specfem::element::dimension_tag DimensionTag> struct Gravity;

/**
 * @brief Globe gravity data precomputed during assembly setup.
 *
 * Point arrays use the medium-local field index returned by
 * @c simulation_field::get_iglob. Radial tables are host data.
 */
template <> struct Gravity<specfem::element::dimension_tag::dim3> {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;

  Gravity() = default;

  /**
   * @brief Prepare gravity for a Globe mesh while spherical points are cached.
   * @param evaluator Configured reference model oracle.
   * @param planet Validated SI planet constants.
   * @param ellipticity Whether to undo ellipticity for the table lookup.
   * @param full_gravity Whether to store the extra scalar for fluid points.
   * @param mesh Assembly mesh with its host spherical coordinate cache.
   * @param element_types Compute-order medium tags.
   * @param fields Assembled fields providing the medium-local point numbering.
   */
  Gravity(const specfem::globe::ModelEvaluator &evaluator,
          const specfem::globe::PlanetConstants &planet, bool ellipticity,
          bool full_gravity, const specfem::assembly::mesh<dimension> &mesh,
          const specfem::assembly::element_types<dimension> &element_types,
          const specfem::assembly::fields<dimension> &fields);

  specfem::globe::ModelEvaluator::GravityTables radial;
  /// @brief Solid gravity vector times reference density, kg/(m^2 s^2).
  Kokkos::View<type_real *[3], Kokkos::LayoutRight> gravity_pre_store;
  /// @brief Solid gravity gradient times density, kg/(m^3 s^2).
  /// Component order is xx, yy, zz, xy, xz, yz.
  Kokkos::View<type_real *[6], Kokkos::LayoutRight> gravity_h;
  /// @brief Acoustic radial direction times -rho*g/kappa, 1/m.
  Kokkos::View<type_real *[3], Kokkos::LayoutRight> acoustic_gravity_pre_store;
  /// @brief Acoustic -rho*g/kappa, 1/m, stored when full gravity is enabled.
  Kokkos::View<type_real *> gravity_rho_g_over_kappa;
};

} // namespace specfem::assembly
