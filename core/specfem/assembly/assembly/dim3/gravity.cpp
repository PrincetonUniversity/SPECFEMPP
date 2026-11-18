#include "specfem/assembly/assembly/dim3/gravity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace specfem::assembly::gravity_impl {

double spherical_radius(
    double radius, double cos_theta,
    const specfem::globe::ModelEvaluator::EllipticitySpline &spline);

} // namespace specfem::assembly::gravity_impl

double specfem::assembly::gravity_impl::spherical_radius(
    const double radius, const double cos_theta,
    const specfem::globe::ModelEvaluator::EllipticitySpline &spline) {
  // Same cubic and endpoint extrapolation as Globe's revert_ellipticity_rtheta.
  const auto &radii = spline.radii;
  const auto upper =
      std::upper_bound(radii.begin() + 1, radii.end() - 1, radius);
  const auto hi = upper - radii.begin();
  const auto lo = hi - 1;
  const double width = radii[hi] - radii[lo];
  const double a = (radii[hi] - radius) / width;
  const double b = (radius - radii[lo]) / width;
  const double ell = a * spline.values[lo] + b * spline.values[hi] +
                     ((a * a * a - a) * spline.second_derivatives[lo] +
                      (b * b * b - b) * spline.second_derivatives[hi]) *
                         (width * width) / 6.0;
  const double p20 = 0.5 * (3.0 * cos_theta * cos_theta - 1.0);
  return radius / (1.0 - (2.0 / 3.0) * ell * p20);
}

specfem::assembly::Gravity<specfem::element::dimension_tag::dim3>::Gravity(
    const specfem::globe::ModelEvaluator &evaluator,
    const specfem::globe::PlanetConstants &planet, const bool ellipticity,
    const bool full_gravity, const specfem::assembly::mesh<dimension> &mesh,
    const specfem::assembly::element_types<dimension> &element_types,
    const specfem::assembly::fields<dimension> &fields)
    : radial(evaluator.gravity_tables()) {
  using Medium = specfem::element::medium_tag;
  using Field = specfem::simulation::field_type;
  const auto &coord = mesh.h_spherical_coord;
  if (mesh.nspec == 0) {
    return;
  }
  if (coord.data() == nullptr) {
    throw std::runtime_error("Gravity setup requires spherical coordinates");
  }

  const auto spline = ellipticity
                          ? evaluator.ellipticity_spline()
                          : specfem::globe::ModelEvaluator::EllipticitySpline{};
  const double planet_radius = planet.r_planet();
  const int table_size = static_cast<int>(radial.density.size());
  const auto populate = [&](const auto &field) {
    const int nsolid = field.template get_nglob<Medium::elastic>();
    const int nfluid = field.template get_nglob<Medium::acoustic>();
    gravity_pre_store = { "gravity_pre_store", nsolid };
    gravity_h = { "gravity_h", nsolid };
    acoustic_gravity_pre_store = { "acoustic_gravity_pre_store", nfluid };
    if (full_gravity) {
      gravity_rho_g_over_kappa = { "gravity_rho_g_over_kappa", nfluid };
    }

    const auto solid_vector = Kokkos::create_mirror_view(gravity_pre_store);
    const auto solid_tensor = Kokkos::create_mirror_view(gravity_h);
    const auto fluid_vector =
        Kokkos::create_mirror_view(acoustic_gravity_pre_store);
    const auto fluid_scalar =
        Kokkos::create_mirror_view(gravity_rho_g_over_kappa);
    std::vector<bool> solid_visited(nsolid);
    std::vector<bool> fluid_visited(nfluid);

    for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
      const auto medium = element_types.get_medium_tag(ispec);
      if (medium != Medium::acoustic && medium != Medium::elastic) {
        throw std::runtime_error("Unsupported Globe medium in gravity setup");
      }
      const bool fluid = medium == Medium::acoustic;
      auto &visited = fluid ? fluid_visited : solid_visited;
      for (int iz = 0; iz < mesh.element_grid.ngllz; ++iz) {
        for (int iy = 0; iy < mesh.element_grid.nglly; ++iy) {
          for (int ix = 0; ix < mesh.element_grid.ngllx; ++ix) {
            const int point =
                fluid ? field.template get_iglob<false, Medium::acoustic>(
                            ispec, iz, iy, ix)
                      : field.template get_iglob<false, Medium::elastic>(
                            ispec, iz, iy, ix);
            if (visited[point]) {
              continue;
            }
            visited[point] = true;

            const double radius = coord(ispec, iz, iy, ix, 0);
            const double theta = coord(ispec, iz, iy, ix, 1);
            const double phi = coord(ispec, iz, iy, ix, 2);
            const double cos_theta = std::cos(theta);
            const double sin_theta = std::sin(theta);
            const std::array<double, 3> direction{ sin_theta * std::cos(phi),
                                                   sin_theta * std::sin(phi),
                                                   cos_theta };
            const double r_table =
                ellipticity ? specfem::assembly::gravity_impl::spherical_radius(
                                  radius, cos_theta, spline)
                            : radius;
            // Globe uses one-based NINT, then clamps to [1, NRAD_GRAVITY].
            const int sample =
                std::clamp(std::lround(r_table / planet_radius /
                                       radial.range_max * table_size),
                           1L, static_cast<long>(table_size)) -
                1;

            if (fluid) {
              const double factor = radial.minus_rho_g_over_kappa_fluid[sample];
              for (int c = 0; c < 3; ++c) {
                fluid_vector(point, c) = factor * direction[c];
              }
              if (full_gravity) {
                fluid_scalar(point) = factor;
              }
              continue;
            }

            // Use the analytic limit inside Globe's 100 m inner-core guard.
            const bool near_center = radius < 100.0;
            const double rho =
                near_center ? radial.center_density : radial.density[sample];
            const double g = near_center ? radial.center_gradient * radius
                                         : radial.minus_gravity[sample];
            const double dg = near_center ? radial.center_gradient
                                          : radial.minus_deriv_gravity[sample];
            const double g_over_r =
                near_center ? radial.center_gradient : g / radius;
            const double slope = dg - g_over_r;
            for (int c = 0; c < 3; ++c) {
              solid_vector(point, c) = rho * g * direction[c];
              solid_tensor(point, c) =
                  rho * (g_over_r + slope * direction[c] * direction[c]);
            }
            solid_tensor(point, 3) = rho * slope * direction[0] * direction[1];
            solid_tensor(point, 4) = rho * slope * direction[0] * direction[2];
            solid_tensor(point, 5) = rho * slope * direction[1] * direction[2];
          }
        }
      }
    }

    Kokkos::deep_copy(gravity_pre_store, solid_vector);
    Kokkos::deep_copy(gravity_h, solid_tensor);
    Kokkos::deep_copy(acoustic_gravity_pre_store, fluid_vector);
    if (full_gravity) {
      Kokkos::deep_copy(gravity_rho_g_over_kappa, fluid_scalar);
    }
  };

  const auto &adjoint = fields.get_simulation_field<Field::adjoint>();
  if (adjoint.h_index_mapping.data() != nullptr) {
    populate(adjoint);
  } else {
    populate(fields.get_simulation_field<Field::forward>());
  }
}
