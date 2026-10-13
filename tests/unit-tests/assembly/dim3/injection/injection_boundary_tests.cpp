#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/boundaries/dim3/impl/injection/extract_injection_points.hpp"
#include "specfem/injection/fk/boundary_coupling.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/solver.hpp"
#include "specfem/injection/fk/time_window.hpp"
#include "specfem/injection/injection_inputs.hpp"
#include "specfem/injection/injection_points.hpp"
#include "specfem/io.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/receivers.hpp"
#include "specfem/source.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

using dim3 = std::integral_constant<specfem::element::dimension_tag,
                                    specfem::element::dimension_tag::dim3>;
constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;

// Half-space material of the InjectionStaceyBox fixture.
constexpr type_real k_rho = 2700;
constexpr type_real k_vp = 3000;
constexpr type_real k_vs = static_cast<type_real>(1732.051);

constexpr type_real k_dt = static_cast<type_real>(0.05);
constexpr int k_nstep = 256;

specfem::injection::fk::LayeredModel make_half_space_model() {
  specfem::injection::fk::ElasticIsotropicLayer half_space;
  half_space.density = k_rho;
  half_space.p_velocity = k_vp;
  half_space.s_velocity = k_vs;
  half_space.thickness = static_cast<type_real>(0);
  return specfem::injection::fk::LayeredModel({}, { half_space });
}

specfem::injection::fk::IncidentWave make_wave() {
  specfem::injection::fk::IncidentWave wave;
  wave.type = specfem::injection::fk::incident_wave_type::p;
  wave.azimuth_phi = static_cast<type_real>(0);
  wave.take_off_theta = static_cast<type_real>(20.0 * M_PI / 180.0);
  wave.origin_x = static_cast<type_real>(0);
  wave.origin_y = static_cast<type_real>(0);
  wave.origin_z = static_cast<type_real>(0);
  wave.origin_time = static_cast<type_real>(5);
  wave.amplitude = static_cast<type_real>(1);
  wave.gaussian_half_duration = static_cast<type_real>(2);
  return wave;
}

specfem::injection::injection_inputs make_inputs() {
  specfem::injection::injection_inputs inputs;
  inputs.enabled = true;
  inputs.method = "fk";
  inputs.model = make_half_space_model();
  inputs.wave = make_wave();
  inputs.derivative = specfem::injection::fk::field_derivative::displacement;
  inputs.frequency_max = static_cast<type_real>(1);
  inputs.frequency_sampling = static_cast<type_real>(2);
  inputs.time_window_length = static_cast<type_real>(40);
  inputs.window_steps = 0;
  inputs.compute_traction = true;
  inputs.consistency_tolerance = 1e-3;
  return inputs;
}

// Build the InjectionStaceyBox assembly with injection enabled.
specfem::assembly::assembly<dim3_tag> make_assembly() {
  auto mesh =
      specfem::io::read_3d_mesh("data/dim3/InjectionStaceyBox/database.bin",
                                specfem::attenuation::Setup{ false });

  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  std::vector<std::shared_ptr<specfem::sources::source<dim3_tag>>> sources;
  std::vector<std::shared_ptr<specfem::receivers::receiver<dim3_tag>>>
      receivers;
  std::vector<specfem::enums::wavefield> seismogram_types = {
    specfem::enums::wavefield::displacement
  };

  return specfem::assembly::assembly<dim3_tag>(
      mesh, quadrature, sources, receivers, seismogram_types,
      static_cast<type_real>(0), k_dt, k_nstep, 1, 1,
      specfem::simulation::type::forward, false, nullptr,
      specfem::element_coupling::flux_scheme_configuration(), make_inputs());
}

// Re-extract the injection points from a built assembly (same code path the
// assembly ctor used, so the row order matches the provider's).
specfem::injection::injection_points
extract(const specfem::assembly::assembly<dim3_tag> &assembly) {
  const auto &tags = assembly.boundaries.stacey.h_quadrature_point_boundary_tag;
  specfem::assembly::boundaries_impl::injection_boundary<
      dim3_tag>::PointIndexView point_index("test_point_index", tags.extent(0),
                                            tags.extent(1), tags.extent(2),
                                            tags.extent(3));
  auto h_point_index = Kokkos::create_mirror_view(point_index);
  return specfem::assembly::boundaries_impl::extract_stacey_injection_points(
      assembly.boundaries.stacey, assembly.boundaries.h_stacey_index_mapping,
      assembly.mesh, assembly.properties, assembly.element_types,
      h_point_index);
}

} // namespace

// ===========================================================================
// Extractor geometry: the enumerated Stacey points carry the right coordinates,
// unit outward normals, medium, and local material.
// ===========================================================================
TEST(InjectionStaceyBox, ExtractorGeometry) {
  const auto assembly = make_assembly();
  const auto points = extract(assembly);

  ASSERT_GT(points.size(), 0);

  const double zmax = static_cast<double>(assembly.mesh.zmax);
  double min_depth = 1e30;
  double max_depth = -1e30;
  for (int i = 0; i < points.size(); ++i) {
    // Medium and material match the homogeneous elastic half-space.
    EXPECT_EQ(points.medium(i), specfem::element::medium_tag::elastic);
    EXPECT_NEAR(static_cast<double>(points.rho(i)), k_rho, 1.0);
    EXPECT_NEAR(static_cast<double>(points.vp(i)), k_vp, 1.0);
    EXPECT_NEAR(static_cast<double>(points.vs(i)), k_vs, 1.0);

    // Outward normal is a unit vector.
    const double nx = static_cast<double>(points.normal_x(i));
    const double ny = static_cast<double>(points.normal_y(i));
    const double nz = static_cast<double>(points.normal_z(i));
    EXPECT_NEAR(std::sqrt(nx * nx + ny * ny + nz * nz), 1.0, 1e-5);

    // Depth is positive-down from the free surface (z = zmax - z_mesh).
    const double depth = static_cast<double>(points.z(i));
    EXPECT_GE(depth, -1e-3);
    EXPECT_LE(depth, 3000.0 + 1e-3);
    min_depth = std::min(min_depth, depth);
    max_depth = std::max(max_depth, depth);
  }

  // Stacey faces span the full depth: the top edge of the lateral faces sits at
  // the free surface (depth 0), the bottom face at the model base (depth 3000).
  EXPECT_NEAR(min_depth, 0.0, 1e-3);
  EXPECT_NEAR(max_depth, 3000.0, 1e-3);
  EXPECT_LT(zmax, 1e-3); // free surface at z = 0
}

// ===========================================================================
// The assembly holds a populated injection boundary whose point-index map is
// consistent with the number of extracted points.
// ===========================================================================
TEST(InjectionStaceyBox, AssemblyHoldsInjection) {
  const auto assembly = make_assembly();
  const auto &injection = assembly.boundaries.injection;

  ASSERT_TRUE(injection.enabled());
  const int expected = extract(assembly).size();
  EXPECT_EQ(injection.number_of_points, expected);

  // The point-index map marks exactly `number_of_points` injection points.
  int marked = 0;
  const auto &map = injection.h_point_index;
  for (std::size_t e = 0; e < map.extent(0); ++e) {
    for (std::size_t iz = 0; iz < map.extent(1); ++iz) {
      for (std::size_t iy = 0; iy < map.extent(2); ++iy) {
        for (std::size_t ix = 0; ix < map.extent(3); ++ix) {
          if (map(e, iz, iy, ix) >= 0) {
            ++marked;
          }
        }
      }
    }
  }
  EXPECT_EQ(marked, injection.number_of_points);
}

// ===========================================================================
// End-to-end: the frame buffer the provider reconstructs at the Stacey points
// matches an independent, self-consistent FK solve on the same points.
// ===========================================================================
TEST(InjectionStaceyBox, EndToEndMatchesDirectSolve) {
  const auto assembly = make_assembly();
  const auto &injection = assembly.boundaries.injection;
  ASSERT_TRUE(injection.enabled());

  const auto points = extract(assembly);
  const int npoints = points.size();
  ASSERT_GT(npoints, 0);

  // Independent reference: convert the same points and run the same FK solve.
  const auto model = make_half_space_model();
  const auto wave = make_wave();
  specfem::injection::fk::TimeWindow window;
  window.dt = k_dt;
  window.nstep = k_nstep;
  window.frequency_max = static_cast<type_real>(1);
  window.frequency_sampling = static_cast<type_real>(2);
  window.time_window_length = static_cast<type_real>(40);

  const auto eval_points = specfem::injection::fk::eval_points_from_points(
      points, model, /*compute_traction=*/true, /*tolerance=*/1e-3);
  const auto result = specfem::injection::fk::solve(
      model, wave, window, eval_points, /*compute_traction=*/true);

  const auto &provider = injection.provider;
  provider->ensure_window(0);
  const auto &buffer = provider->frame_buffer();
  const int ncomp = buffer.number_of_components();
  const int W = buffer.number_of_steps();
  ASSERT_EQ(ncomp, 6);

  Kokkos::View<type_real ***> reference("reference", npoints, ncomp, W);
  result.sample(0, W, reference);
  auto h_reference = Kokkos::create_mirror_view(reference);
  Kokkos::deep_copy(h_reference, reference);

  auto h_buffer = Kokkos::create_mirror_view(buffer.values());
  Kokkos::deep_copy(h_buffer, buffer.values());

  double max_difference = 0.0;
  double max_reference = 0.0;
  double max_displacement = 0.0;
  for (int ip = 0; ip < npoints; ++ip) {
    for (int ic = 0; ic < ncomp; ++ic) {
      for (int k = 0; k < W; ++k) {
        const double got = static_cast<double>(h_buffer(ip, ic, k));
        const double ref = static_cast<double>(h_reference(ip, ic, k));
        EXPECT_TRUE(std::isfinite(got));
        max_difference = std::max(max_difference, std::fabs(got - ref));
        max_reference = std::max(max_reference, std::fabs(ref));
        if (ic < 3) {
          max_displacement = std::max(max_displacement, std::fabs(got));
        }
      }
    }
  }

  EXPECT_GT(max_displacement, 1e-30) << "displacement components are all-zero";
  EXPECT_LE(max_difference, 1e-5 * (1.0 + max_reference))
      << "frame buffer disagrees with the direct FK solve";
}
