#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh/mesh_base.hpp"
#include "specfem/mesh_entity.hpp"
#include <Kokkos_Core.hpp>
#include <array>

namespace specfem {
namespace mesh {

/**
 * @brief How the mesher defined the injection surface
 *
 * Values match the injection kind in the database boundary header.
 */
enum class InjectionSurfaceKind : int { none = 0, box_sides = 1 };

/**
 * @brief 3D wavefield-injection face information
 *
 * @c side is +1 when the face normal points out of the total-field region,
 * which is always the case for box sides; -1 is reserved for interior
 * injection surfaces.
 */
template <> struct InjectionSurface<specfem::element::dimension_tag::dim3> {

  constexpr static auto dimension =
      specfem::element::dimension_tag::dim3; ///< Dimension type

  constexpr static int nsides = 6;

  InjectionSurfaceKind kind = InjectionSurfaceKind::none;

  /// xmin, xmax, ymin, ymax, bottom, top
  std::array<bool, nsides> sides = {};

  int nfaces = 0;

  Kokkos::View<int *, Kokkos::HostSpace> index_mapping;

  Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace> type;

  Kokkos::View<int *, Kokkos::HostSpace> side;

  /**
   * @name Constructors
   */
  ///@{
  InjectionSurface() = default;

  /**
   * @brief Allocate storage for @p nfaces injection faces.
   *
   * @param kind How the mesher defined the surface
   * @param sides Sides of the box that carry injection faces
   * @param nfaces Number of injection faces
   */
  InjectionSurface(const InjectionSurfaceKind kind,
                   const std::array<bool, nsides> &sides, const int nfaces);
  ///@}
};

} // namespace mesh
} // namespace specfem
