#pragma once

/**
 * @brief Device add-on that injects the incident field at Stacey boundary
 * points.
 *
 * Applied at the Stacey apply seam of the stiffness kernel, right after the
 * absorbing boundary condition. The call site is uniform for every boundary
 * tag, medium, and wavefield; all dispatch lives here. Structured like the
 * Stacey boundary condition (@c boundary_conditions/stacey): a thin public
 * @c apply_injection template that gates and delegates to small per-medium
 * @c impl_injection functions (non-SIMD and SIMD overloads).
 */

#include "specfem/datatype.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/injection/frame_layout.hpp"
#include "specfem/setup.hpp"
#include "specfem/simulation.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem {
namespace injection {
namespace impl {

using elastic_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::elastic>;
using acoustic_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::acoustic>;
using isotropic_type =
    std::integral_constant<specfem::element::property_tag,
                           specfem::element::property_tag::isotropic>;

/**
 * @brief Frame-buffer row for a scalar quadrature point, or -1 if not injected.
 */
template <typename InjectionBoundaryType, typename IndexType>
KOKKOS_FORCEINLINE_FUNCTION int
injection_row(const InjectionBoundaryType &injection, const IndexType &index) {
  const int local = injection.stacey_index_mapping(index.ispec);
  if (local < 0)
    return -1;
  return injection.point_of(local, index.iz, index.iy, index.ix);
}

/**
 * @brief Per-lane frame-buffer rows for a SIMD quadrature point (-1 where
 * none).
 *
 * Lanes are consecutive elements at the same GLL node; an injection row exists
 * only on true Stacey points, so inactive and non-injection lanes stay -1.
 */
template <typename InjectionBoundaryType, typename IndexType>
KOKKOS_FORCEINLINE_FUNCTION void
injection_rows(const InjectionBoundaryType &injection, const IndexType &index,
               int *row, const int lanes) {
  for (int lane = 0; lane < lanes; ++lane) {
    row[lane] = -1;
    if (index.mask(lane)) {
      const int local = injection.stacey_index_mapping(index.ispec + lane);
      if (local >= 0)
        row[lane] = injection.point_of(local, index.iz, index.iy, index.ix);
    }
  }
}

// ---------------------------------------------------------------------------
// Elastic injection: accel += (t* + Z(v*)) * Jw, with Z the Stacey impedance
// operator (see the apply_injection doc comment). Parameterization-agnostic:
// isotropic and anisotropic share the Voigt impedances rho_vp, rho_vs.
// ---------------------------------------------------------------------------

template <
    specfem::element::property_tag PropertyTag, typename IndexType,
    typename InjectionBoundaryType, typename PointBoundaryType,
    typename PointPropertyType, typename PointAccelerationType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_injection(
    const elastic_type &,
    const std::integral_constant<specfem::element::property_tag, PropertyTag> &,
    const IndexType &index, const int istep,
    const InjectionBoundaryType &injection,
    const PointBoundaryType &point_boundary,
    const PointPropertyType &point_property,
    PointAccelerationType &acceleration) {

  const int row = injection_row(injection, index);
  if (row < 0)
    return;

  namespace frame = specfem::injection::frame_component;
  const auto &fb = injection.frame;
  const auto &face_normal = point_boundary.face_normal;
  const type_real jacobian2d = face_normal.l2_norm();
  const type_real inv_jacobian2d = static_cast<type_real>(1) / jacobian2d;
  const type_real nhat[3] = { face_normal(0) * inv_jacobian2d,
                              face_normal(1) * inv_jacobian2d,
                              face_normal(2) * inv_jacobian2d };
  const type_real jw = jacobian2d * point_boundary.face_weight;

  const type_real vstar[3] = {
    fb.load_on_device(istep, row, frame::displacement_x),
    fb.load_on_device(istep, row, frame::displacement_y),
    fb.load_on_device(istep, row, frame::displacement_z)
  };
  const type_real tstar[3] = { fb.load_on_device(istep, row, frame::traction_x),
                               fb.load_on_device(istep, row, frame::traction_y),
                               fb.load_on_device(istep, row,
                                                 frame::traction_z) };

  const type_real vn =
      vstar[0] * nhat[0] + vstar[1] * nhat[1] + vstar[2] * nhat[2];
  const type_real rho_vp_minus_vs =
      point_property.rho_vp() - point_property.rho_vs();
  const type_real rho_vs = point_property.rho_vs();
  for (int comp = 0; comp < 3; ++comp) {
    const type_real z =
        vn * nhat[comp] * rho_vp_minus_vs + vstar[comp] * rho_vs;
    acceleration(comp) += (tstar[comp] + z) * jw;
  }
}

template <
    specfem::element::property_tag PropertyTag, typename IndexType,
    typename InjectionBoundaryType, typename PointBoundaryType,
    typename PointPropertyType, typename PointAccelerationType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_injection(
    const elastic_type &,
    const std::integral_constant<specfem::element::property_tag, PropertyTag> &,
    const IndexType &index, const int istep,
    const InjectionBoundaryType &injection,
    const PointBoundaryType &point_boundary,
    const PointPropertyType &point_property,
    PointAccelerationType &acceleration) {

  using datatype = typename PointBoundaryType::simd::datatype;
  using mask_type = typename PointBoundaryType::simd::mask_type;
  constexpr int lanes = static_cast<int>(mask_type::size());

  int row[lanes];
  injection_rows(injection, index, row, lanes);
  const mask_type inject_mask([&](std::size_t lane) { return row[lane] >= 0; });
  if (Kokkos::Experimental::none_of(inject_mask))
    return;

  namespace frame = specfem::injection::frame_component;
  const auto &fb = injection.frame;
  const auto &face_normal = point_boundary.face_normal;
  const auto jacobian2d = face_normal.l2_norm();
  const auto inv_jacobian2d = static_cast<type_real>(1) / jacobian2d;
  const datatype nhat[3] = { face_normal(0) * inv_jacobian2d,
                             face_normal(1) * inv_jacobian2d,
                             face_normal(2) * inv_jacobian2d };
  const auto jw = jacobian2d * point_boundary.face_weight;

  const datatype vstar[3] = {
    datatype([&](std::size_t lane) {
      return row[lane] >= 0
                 ? fb.load_on_device(istep, row[lane], frame::displacement_x)
                 : static_cast<type_real>(0);
    }),
    datatype([&](std::size_t lane) {
      return row[lane] >= 0
                 ? fb.load_on_device(istep, row[lane], frame::displacement_y)
                 : static_cast<type_real>(0);
    }),
    datatype([&](std::size_t lane) {
      return row[lane] >= 0
                 ? fb.load_on_device(istep, row[lane], frame::displacement_z)
                 : static_cast<type_real>(0);
    })
  };
  const datatype tstar[3] = {
    datatype([&](std::size_t lane) {
      return row[lane] >= 0
                 ? fb.load_on_device(istep, row[lane], frame::traction_x)
                 : static_cast<type_real>(0);
    }),
    datatype([&](std::size_t lane) {
      return row[lane] >= 0
                 ? fb.load_on_device(istep, row[lane], frame::traction_y)
                 : static_cast<type_real>(0);
    }),
    datatype([&](std::size_t lane) {
      return row[lane] >= 0
                 ? fb.load_on_device(istep, row[lane], frame::traction_z)
                 : static_cast<type_real>(0);
    })
  };

  const datatype vn =
      vstar[0] * nhat[0] + vstar[1] * nhat[1] + vstar[2] * nhat[2];
  const auto rho_vp_minus_vs =
      point_property.rho_vp() - point_property.rho_vs();
  const auto rho_vs = point_property.rho_vs();
  for (int comp = 0; comp < 3; ++comp) {
    const datatype z = vn * nhat[comp] * rho_vp_minus_vs + vstar[comp] * rho_vs;
    const datatype delta = (tstar[comp] + z) * jw;
    acceleration(comp) = Kokkos::Experimental::condition(
        inject_mask, acceleration(comp) + delta, acceleration(comp));
  }
}

// ---------------------------------------------------------------------------
// Acoustic: potential_dot_dot += Jw·(u*·n̂) + Jw·(1/ρvp)·χ̇*, injecting the
// incident fluid displacement and potential-rate (SPECFEM3D
// compute_stacey_acoustic).
// ---------------------------------------------------------------------------

template <
    typename IndexType, typename InjectionBoundaryType,
    typename PointBoundaryType, typename PointPropertyType,
    typename PointAccelerationType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_injection(const acoustic_type &, const isotropic_type &,
               const IndexType &index, const int istep,
               const InjectionBoundaryType &injection,
               const PointBoundaryType &point_boundary,
               const PointPropertyType &point_property,
               PointAccelerationType &acceleration) {

  const int row = injection_row(injection, index);
  if (row < 0)
    return;

  namespace frame = specfem::injection::frame_component;
  const auto &fb = injection.frame;
  const auto &face_normal = point_boundary.face_normal;
  const type_real jacobian2d = face_normal.l2_norm();
  const type_real inv_jacobian2d = static_cast<type_real>(1) / jacobian2d;
  const type_real jw = jacobian2d * point_boundary.face_weight;

  // Incident normal displacement u*·n̂ and potential-rate χ̇*.
  const type_real un =
      (fb.load_on_device(istep, row, frame::displacement_x) * face_normal(0) +
       fb.load_on_device(istep, row, frame::displacement_y) * face_normal(1) +
       fb.load_on_device(istep, row, frame::displacement_z) * face_normal(2)) *
      inv_jacobian2d;
  const type_real chi_dot =
      fb.load_on_device(istep, row, frame::potential_rate);

  acceleration(0) += jw * (un + point_property.rho_vpinverse() * chi_dot);
}

template <
    typename IndexType, typename InjectionBoundaryType,
    typename PointBoundaryType, typename PointPropertyType,
    typename PointAccelerationType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
impl_injection(const acoustic_type &, const isotropic_type &,
               const IndexType &index, const int istep,
               const InjectionBoundaryType &injection,
               const PointBoundaryType &point_boundary,
               const PointPropertyType &point_property,
               PointAccelerationType &acceleration) {

  using datatype = typename PointBoundaryType::simd::datatype;
  using mask_type = typename PointBoundaryType::simd::mask_type;
  constexpr int lanes = static_cast<int>(mask_type::size());

  int row[lanes];
  injection_rows(injection, index, row, lanes);
  const mask_type inject_mask([&](std::size_t lane) { return row[lane] >= 0; });
  if (Kokkos::Experimental::none_of(inject_mask))
    return;

  namespace frame = specfem::injection::frame_component;
  const auto &fb = injection.frame;
  const auto &face_normal = point_boundary.face_normal;
  const auto jacobian2d = face_normal.l2_norm();
  const auto inv_jacobian2d = static_cast<type_real>(1) / jacobian2d;
  const auto jw = jacobian2d * point_boundary.face_weight;

  const datatype ux([&](std::size_t lane) {
    return row[lane] >= 0
               ? fb.load_on_device(istep, row[lane], frame::displacement_x)
               : static_cast<type_real>(0);
  });
  const datatype uy([&](std::size_t lane) {
    return row[lane] >= 0
               ? fb.load_on_device(istep, row[lane], frame::displacement_y)
               : static_cast<type_real>(0);
  });
  const datatype uz([&](std::size_t lane) {
    return row[lane] >= 0
               ? fb.load_on_device(istep, row[lane], frame::displacement_z)
               : static_cast<type_real>(0);
  });
  const datatype chi_dot([&](std::size_t lane) {
    return row[lane] >= 0
               ? fb.load_on_device(istep, row[lane], frame::potential_rate)
               : static_cast<type_real>(0);
  });

  const datatype un =
      (ux * face_normal(0) + uy * face_normal(1) + uz * face_normal(2)) *
      inv_jacobian2d;
  const datatype delta = jw * (un + point_property.rho_vpinverse() * chi_dot);
  acceleration(0) = Kokkos::Experimental::condition(
      inject_mask, acceleration(0) + delta, acceleration(0));
}

} // namespace impl

/**
 * @brief Add the incident injection field to the boundary acceleration.
 *
 * A compile-time no-op unless the quadrature point sits on a Stacey-derived
 * boundary of a 3D mesh and the wavefield is the forward field. For injection
 * points it delegates by medium to @c impl::impl_injection, which adds the
 * incident traction and the Stacey impedance applied to the incident velocity.
 * Combined with the absorbing term \f$-Z(v)Jw\f$, the boundary then absorbs the
 * scattered field \f$v-v^*\f$ and injects the incident traction.
 *
 * Takes the whole boundary container and reaches @c boundaries.injection only
 * inside the dim3 branch, so the dim2 container needs no injection member.
 *
 * @tparam WavefieldTag Simulation field type; injection fires only for the
 *                      forward field, so it never contaminates an adjoint run.
 * @tparam IndexType Quadrature-point index type (scalar or SIMD).
 * @tparam BoundariesType Assembly boundary container type.
 * @tparam PointBoundaryType Point boundary type (carries the Stacey geometry).
 * @tparam PointPropertyType Point property type (carries the impedance).
 * @tparam PointAccelerationType Point acceleration type (modified in place).
 * @param index Quadrature-point index.
 * @param istep Current forward time step.
 * @param boundaries Assembly boundary container (holds the injection state).
 * @param point_boundary Loaded Stacey boundary data at this point.
 * @param point_property Loaded material properties at this point.
 * @param acceleration Acceleration accumulator for this point.
 */
template <specfem::simulation::field_type WavefieldTag, typename IndexType,
          typename BoundariesType, typename PointBoundaryType,
          typename PointPropertyType, typename PointAccelerationType>
KOKKOS_FORCEINLINE_FUNCTION void
apply_injection(const IndexType &index, const int istep,
                const BoundariesType &boundaries,
                const PointBoundaryType &point_boundary,
                const PointPropertyType &point_property,
                PointAccelerationType &acceleration) {

  constexpr auto boundary_tag = PointBoundaryType::boundary_tag;

  if constexpr (PointBoundaryType::dimension_tag ==
                    specfem::element::dimension_tag::dim3 &&
                WavefieldTag == specfem::simulation::field_type::forward &&
                (boundary_tag == specfem::element::boundary_tag::stacey ||
                 boundary_tag == specfem::element::boundary_tag::
                                     composite_stacey_dirichlet)) {

    const auto &injection = boundaries.injection;
    if (!injection.enabled())
      return;

    specfem::injection::impl::impl_injection(
        std::integral_constant<specfem::element::medium_tag,
                               PointPropertyType::medium_tag>{},
        std::integral_constant<specfem::element::property_tag,
                               PointPropertyType::property_tag>{},
        index, istep, injection, point_boundary, point_property, acceleration);
  }
}

} // namespace injection
} // namespace specfem
