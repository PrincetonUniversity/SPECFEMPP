#include "specfem/injection/fk/impl/driver.hpp"

#include "specfem/injection/fk/impl/layer_stack.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

specfem::injection::fk::FkResult specfem::injection::fk_impl::make_result(
    const specfem::injection::fk_impl::DepthSeries &series,
    const specfem::injection::fk_impl::DepthGroups &groups,
    const specfem::injection::fk::EvalPoints &points,
    const specfem::injection::fk_impl::FkSetup &setup, double delay_stretch) {
  const int ngroups = series.number_of_groups;
  const int nseries = series.number_of_series;
  const int ncoef = series.coefficient_count;
  const int npoints = points.size();

  // Series coefficients: host double -> device type_real, LayoutRight.
  Kokkos::View<type_real ***, Kokkos::LayoutRight> coefficients(
      "fk_series_coefficients", ngroups, nseries, ncoef);
  auto host_coefficients = Kokkos::create_mirror_view(coefficients);
  for (int g = 0; g < ngroups; ++g) {
    for (int s = 0; s < nseries; ++s) {
      for (int k = 0; k < ncoef; ++k) {
        host_coefficients(g, s, k) =
            static_cast<type_real>(series.coefficients(g, s, k));
      }
    }
  }
  Kokkos::deep_copy(coefficients, host_coefficients);

  // Per-point group index.
  Kokkos::View<int *> point_group("fk_point_group", npoints);
  auto host_point_group = Kokkos::create_mirror_view(point_group);
  // Per-point abscissa offset pad - tau/dt_min.
  Kokkos::View<double *> point_offset("fk_point_offset", npoints);
  auto host_point_offset = Kokkos::create_mirror_view(point_offset);
  // Per-point traction material: xi1, xim, bulk, nx, ny, nz.
  Kokkos::View<type_real **, Kokkos::LayoutRight> point_material(
      "fk_point_material", npoints, 6);
  auto host_point_material = Kokkos::create_mirror_view(point_material);
  // Per-point acoustic flag.
  Kokkos::View<int *> point_is_acoustic("fk_point_is_acoustic", npoints);
  auto host_point_is_acoustic = Kokkos::create_mirror_view(point_is_acoustic);

  // Optional per-point Lame factors and normals are host-resident Views.
  const bool have_lame = points.lame_ratio_xi1.extent(0) > 0;
  const bool have_normal = points.normal_x.extent(0) > 0;

  const double dt_min = setup.dt_min;
  const double pad = static_cast<double>(series.pad_samples);

  for (int i = 0; i < npoints; ++i) {
    const int g = groups.point_group[i];
    host_point_group(i) = g;
    host_point_offset(i) = specfem::injection::fk_impl::point_abscissa_offset(
        pad, groups.time_delay[i], delay_stretch, dt_min);

    const bool acoustic = (groups.groups[g].medium ==
                           specfem::injection::fk::point_medium::acoustic);
    host_point_is_acoustic(i) = acoustic ? 1 : 0;

    host_point_material(i, 0) =
        have_lame ? points.lame_ratio_xi1(i) : static_cast<type_real>(0);
    host_point_material(i, 1) =
        have_lame ? points.lame_ratio_xim(i) : static_cast<type_real>(0);
    host_point_material(i, 2) =
        have_lame ? points.lame_ratio_bulk(i) : static_cast<type_real>(0);
    host_point_material(i, 3) =
        have_normal ? points.normal_x(i) : static_cast<type_real>(0);
    host_point_material(i, 4) =
        have_normal ? points.normal_y(i) : static_cast<type_real>(0);
    host_point_material(i, 5) =
        have_normal ? points.normal_z(i) : static_cast<type_real>(0);
  }

  Kokkos::deep_copy(point_group, host_point_group);
  Kokkos::deep_copy(point_offset, host_point_offset);
  Kokkos::deep_copy(point_material, host_point_material);
  Kokkos::deep_copy(point_is_acoustic, host_point_is_acoustic);

  const type_real cos_phi = static_cast<type_real>(std::cos(setup.phi));
  const type_real sin_phi = static_cast<type_real>(std::sin(setup.phi));

  return specfem::injection::fk::FkResult(
      coefficients, point_group, point_offset, point_material,
      point_is_acoustic, cos_phi, sin_phi, setup.sizes.resampling_rate,
      setup.compute_traction);
}

specfem::injection::fk::FkResult specfem::injection::fk_impl::run_fk_driver(
    const specfem::injection::fk::LayeredModel &model,
    const specfem::injection::fk::IncidentWave &wave,
    const specfem::injection::fk::TimeWindow &window,
    const specfem::injection::fk::EvalPoints &points, bool compute_traction,
    specfem::injection::fk::field_derivative derivative,
    bool reproduce_specfem3d_reference) {
  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);

  // Largest point depth magnitude, for the float round-off grouping tolerance.
  double max_depth = 0.0;
  const int number_of_points = points.size();
  for (int i = 0; i < number_of_points; ++i) {
    max_depth =
        std::max(max_depth, std::fabs(static_cast<double>(points.z(i))));
  }

  const double tolerance =
      specfem::injection::fk_impl::depth_tolerance(max_depth);
  const specfem::injection::fk_impl::DepthGroups groups =
      specfem::injection::fk_impl::group_by_depth(points, model, wave, p,
                                                  tolerance);
  const specfem::injection::fk_impl::FkSetup setup =
      specfem::injection::fk_impl::prepare(model, wave, window, p, groups,
                                           derivative, compute_traction);

  // Delay scaling: 1 in production (exact plane-wave delay); N/(N-1) only when
  // reproducing the SPECFEM3D reference, whose frequency-grid stretched delays.
  const double delay_stretch =
      reproduce_specfem3d_reference
          ? static_cast<double>(setup.sizes.number_of_fft_samples) /
                static_cast<double>(setup.sizes.number_of_fft_samples - 1)
          : 1.0;

  // Window coverage: every SEM step for every point must map to an abscissa in
  // [0, ncoef-1]; spline evaluation clamps out-of-range values into a silent
  // DC offset otherwise. The guard uses the production (un-stretched) delay:
  // the SPECFEM3D-reference mode may ride the window edge by up to the N/(N-1)
  // stretch and rely on the same spline clamp the reference does.
  const int ncoef = setup.sizes.number_of_stored_frequencies;
  const double resampling = static_cast<double>(setup.sizes.resampling_rate);
  const int npoints = points.size();
  for (int i = 0; i < npoints; ++i) {
    const double offset = specfem::injection::fk_impl::point_abscissa_offset(
        static_cast<double>(setup.pad_samples), groups.time_delay[i],
        /*delay_stretch=*/1.0, setup.dt_min);
    const double max_abscissa =
        static_cast<double>(window.nstep - 1) / resampling + offset;
    if (offset < 0.0 || max_abscissa > static_cast<double>(ncoef - 1)) {
      std::ostringstream message;
      message << "run_fk_driver: sampling abscissa out of range for "
                 "point "
              << i << " (offset " << offset << ", max " << max_abscissa
              << ", coefficients " << ncoef
              << "); increase the injection time-window length";
      throw std::runtime_error(message.str());
    }
  }

  const specfem::injection::fk_impl::InterfaceStates states =
      specfem::injection::fk_impl::solve_layer_stack(model, setup);
  const specfem::injection::fk_impl::DepthSeries series =
      specfem::injection::fk_impl::depth_series(groups, states, model, setup,
                                                reproduce_specfem3d_reference);
  return specfem::injection::fk_impl::make_result(series, groups, points, setup,
                                                  delay_stretch);
}
