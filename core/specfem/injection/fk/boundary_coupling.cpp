#include "specfem/injection/fk/boundary_coupling.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

specfem::injection::fk::point_medium
specfem::injection::fk_coupling_impl::resolve_medium_and_check(
    const specfem::injection::injection_points &points, int i,
    const specfem::injection::fk::LayeredModel &model, double tolerance) {
  // Map the generic medium tag to the FK medium.
  specfem::injection::fk::point_medium medium;
  switch (points.medium(i)) {
  case specfem::element::medium_tag::elastic:
    medium = specfem::injection::fk::point_medium::elastic;
    break;
  case specfem::element::medium_tag::acoustic:
    medium = specfem::injection::fk::point_medium::acoustic;
    break;
  default:
    std::ostringstream message;
    message
        << "eval_points_from_points: unsupported medium for injection point "
        << i << "; only elastic and acoustic media are supported";
    throw std::runtime_error(message.str());
  }

  // Look up the 1-D model layer at this depth and compare its material to the
  // mesh material carried by the point.
  const double depth = static_cast<double>(points.z(i));
  const double locate_tolerance = std::max(1e-3, 1e-6 * std::fabs(depth));
  const specfem::injection::fk::LayerLocation location =
      model.locate(depth, medium, locate_tolerance);

  double rho_model = 0.0;
  double vp_model = 0.0;
  double vs_model = 0.0;
  if (medium == specfem::injection::fk::point_medium::acoustic) {
    const specfem::injection::fk::AcousticLayer &layer =
        model.fluid_layers()[location.layer];
    rho_model = static_cast<double>(layer.density);
    vp_model = static_cast<double>(layer.p_velocity);
  } else {
    const specfem::injection::fk::ElasticIsotropicLayer &layer =
        model.elastic_layers()[location.layer];
    rho_model = static_cast<double>(layer.density);
    vp_model = static_cast<double>(layer.p_velocity);
    vs_model = static_cast<double>(layer.s_velocity);
  }

  const double rho_mesh = static_cast<double>(points.rho(i));
  const double vp_mesh = static_cast<double>(points.vp(i));
  const double vs_mesh = static_cast<double>(points.vs(i));

  const auto relative_difference = [](double a, double b) {
    const double scale = std::max(std::fabs(a), std::fabs(b));
    const double difference = std::fabs(a - b);
    return (scale > 0.0) ? difference / scale : difference;
  };

  const auto matches = [&](double rho_layer, double vp_layer, double vs_layer,
                           bool check_vs) {
    return relative_difference(rho_mesh, rho_layer) <= tolerance &&
           relative_difference(vp_mesh, vp_layer) <= tolerance &&
           (!check_vs || relative_difference(vs_mesh, vs_layer) <= tolerance);
  };

  bool matched =
      matches(rho_model, vp_model, vs_model,
              medium == specfem::injection::fk::point_medium::elastic);

  if (!matched) {
    // Mesh GLL points ON a layer interface are shared by elements of both
    // sides and carry their own element's material; accept a match against
    // either layer adjacent to an interface within the locate tolerance. The
    // stored series are interface-continuous, so either side evaluates the
    // same field; the discontinuous moduli enter via the per-point Lame
    // ratios.
    const auto &tops = model.layer_top_depths();
    const int nfluid = model.number_of_fluid_layers();
    for (std::size_t b = 1; b < tops.size() && !matched; ++b) {
      if (std::fabs(depth - static_cast<double>(tops[b])) > locate_tolerance) {
        continue;
      }
      const int adjacent[2] = { static_cast<int>(b) - 1, static_cast<int>(b) };
      for (const int global_layer : adjacent) {
        const bool layer_is_fluid = global_layer < nfluid;
        if (layer_is_fluid !=
            (medium == specfem::injection::fk::point_medium::acoustic)) {
          continue;
        }
        if (layer_is_fluid) {
          const specfem::injection::fk::AcousticLayer &layer =
              model.fluid_layers()[global_layer];
          matched = matches(static_cast<double>(layer.density),
                            static_cast<double>(layer.p_velocity), 0.0, false);
        } else {
          const specfem::injection::fk::ElasticIsotropicLayer &layer =
              model.elastic_layers()[global_layer - nfluid];
          matched = matches(static_cast<double>(layer.density),
                            static_cast<double>(layer.p_velocity),
                            static_cast<double>(layer.s_velocity), true);
        }
        if (matched) {
          break;
        }
      }
    }
  }

  if (!matched) {
    std::ostringstream message;
    message << "eval_points_from_points: injection point " << i << " at depth "
            << depth << " m has mesh material (rho, vp, vs) = (" << rho_mesh
            << ", " << vp_mesh << ", " << vs_mesh
            << ") that disagrees with the 1-D injection model layer ("
            << rho_model << ", " << vp_model << ", " << vs_model
            << ") beyond the relative tolerance " << tolerance
            << ". The injected wavefield is defined by the 1-D model; the mesh "
               "medium at the injection boundary must match it.";
    throw std::runtime_error(message.str());
  }

  return medium;
}

specfem::injection::fk::EvalPoints
specfem::injection::fk::eval_points_from_points(
    const specfem::injection::injection_points &points,
    const specfem::injection::fk::LayeredModel &model, bool compute_traction,
    double tolerance) {
  const int n = points.size();

  specfem::injection::fk::EvalPoints eval_points;
  // Geometry is shared with the input (read-only during the solve).
  eval_points.x = points.x;
  eval_points.y = points.y;
  eval_points.z = points.z;

  eval_points.medium =
      Kokkos::View<specfem::injection::fk::point_medium *, Kokkos::HostSpace>(
          "fk_eval_medium", n);

  const bool have_normals = points.normal_x.extent(0) > 0;
  const bool fill_traction = compute_traction && have_normals;
  if (fill_traction) {
    // The synthesized traction is the textbook sigma.n of the OUTWARD face
    // normal (same as SPECFEM3D's abs_boundary_normal convention): incoming
    // field components then pair constructively with the Stacey impedance
    // Z(v*) while outgoing components cancel — exactly the split the injected
    // boundary needs. See fortran-oracle
    // InjectedTractionConsistentWithImpedance.
    eval_points.normal_x = points.normal_x;
    eval_points.normal_y = points.normal_y;
    eval_points.normal_z = points.normal_z;
    eval_points.lame_ratio_xi1 =
        Kokkos::View<type_real *, Kokkos::HostSpace>("fk_eval_xi1", n);
    eval_points.lame_ratio_xim =
        Kokkos::View<type_real *, Kokkos::HostSpace>("fk_eval_xim", n);
    eval_points.lame_ratio_bulk =
        Kokkos::View<type_real *, Kokkos::HostSpace>("fk_eval_bulk", n);
  }

  for (int i = 0; i < n; ++i) {
    const specfem::injection::fk::point_medium medium =
        specfem::injection::fk_coupling_impl::resolve_medium_and_check(
            points, i, model, tolerance);
    eval_points.medium(i) = medium;

    if (fill_traction) {
      if (medium == specfem::injection::fk::point_medium::elastic) {
        const double vp = static_cast<double>(points.vp(i));
        const double vs = static_cast<double>(points.vs(i));
        const double rho = static_cast<double>(points.rho(i));
        const double mu = rho * vs * vs;
        const double lambda = rho * vp * vp - 2.0 * mu;
        // Matches SPECFEM3D couple_with_injection.f90: xi = mu/(kappa+4/3 mu) =
        // vs^2/vp^2; the stress series uses sigma_rr = xi1*sigma_zz - 4*xim*dx
        // with xim carrying the shear modulus (dx is a strain-like term).
        const double xi = (vs * vs) / (vp * vp);
        eval_points.lame_ratio_xi1(i) = static_cast<type_real>(1.0 - 2.0 * xi);
        eval_points.lame_ratio_xim(i) = static_cast<type_real>((1.0 - xi) * mu);
        eval_points.lame_ratio_bulk(i) =
            static_cast<type_real>(lambda / (2.0 * (lambda + mu)));
      } else {
        eval_points.lame_ratio_xi1(i) = static_cast<type_real>(0);
        eval_points.lame_ratio_xim(i) = static_cast<type_real>(0);
        eval_points.lame_ratio_bulk(i) = static_cast<type_real>(0);
      }
    }
  }

  return eval_points;
}
