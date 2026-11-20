#include "specfem/assembly/assembly/dim3/globe_properties.hpp"
#include "globe_model_evaluator.h"
#include "specfem/globe/model_evaluator.hpp"
#include "specfem/globe/region_codes.hpp"
#include "specfem/io.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace specfem::assembly::globe_properties_test_impl {

constexpr auto dimension = specfem::element::dimension_tag::dim3;
using Medium = specfem::element::medium_tag;
using Property = specfem::element::property_tag;
using Region = specfem::element::region_tag;
using Assembly = specfem::assembly::assembly<dimension>;
using Index = specfem::point::index<dimension, false>;
constexpr double tolerance = std::is_same_v<type_real, double> ? 1.e-10 : 2.e-6;

// Only construct the geometry and containers consumed by the property builder.
// The fixture includes ellipticity, so final and reference coordinates differ.
struct GlobeFixture {
  specfem::mesh::globe3d_mesh mesh;
  Assembly assembly;

  explicit GlobeFixture(const bool anisotropic)
      : mesh(specfem::io::read_globe_mesh(
            "data/dim3_globe/GlobalSmallMesh/DATABASES_MPI/"
            "proc000000_specfempp_database.bin",
            specfem::attenuation::Setup{})) {
    if (anisotropic) {
      mesh.globe.model_config.model_name = "1D_transversely_isotropic_prem";
      // This variant intentionally changes the fixture's resolved model.
      mesh.globe.model_verification.codes.clear();
      mesh.globe.model_verification.flags.clear();
      for (int i = 0; i < mesh.nspec; ++i) {
        if (mesh.globe.element_context[i].region == Region::crust_mantle) {
          mesh.tags.tags_container(i).property_tag = Property::anisotropic;
        }
      }
    }
    const specfem::quadrature::quadratures quadrature(
        specfem::quadrature::gll::gll{});
    assembly.mesh = { mesh.nspec,
                      mesh.control_nodes.ngnod,
                      mesh.element_grid.ngllz,
                      mesh.element_grid.nglly,
                      mesh.element_grid.ngllx,
                      mesh.tags,
                      mesh.adjacency_graph,
                      mesh.control_nodes,
                      quadrature,
                      mesh.globe.reference_coordinates };
    assembly.mesh.compute_spherical_coordinates();
    assembly.element_types = { mesh.nspec, assembly.mesh.element_grid,
                               assembly.mesh, mesh.tags,
                               mesh.globe.element_context };
    assembly.properties = { assembly.element_types, assembly.mesh,
                            mesh.materials, true };
  }
};

template <Medium MediumTag, Property PropertyTag>
void check_properties(const GlobeFixture &fixture,
                      const specfem::globe::ModelEvaluator &oracle,
                      std::ofstream &profile) {
  using Point = specfem::point::properties<
      specfem::tags::Tags<dimension, MediumTag, PropertyTag, false>>;
  const auto &assembly = fixture.assembly;
  const auto &types = assembly.element_types;
  const auto elements = types.get_elements_on_host(MediumTag, PropertyTag);
  const auto properties = assembly.properties;
  const int nx = assembly.mesh.element_grid.ngllx;
  const int ny = assembly.mesh.element_grid.nglly;
  const int nz = assembly.mesh.element_grid.ngllz;
  const int npoints = nx * ny * nz;
  ASSERT_GT(elements.extent(0), 0);

  // Load on device, then compare against independently loaded host properties.
  Kokkos::View<Point *> device_points("globe_device_properties",
                                      elements.extent(0) * npoints);
  const int begin = elements.begin_index();
  Kokkos::parallel_for(
      "check_globe_properties", device_points.extent(0),
      KOKKOS_LAMBDA(const int i) {
        const int p = i % npoints;
        const Index index(begin + i / npoints, p / (nx * ny), (p / nx) % ny,
                          p % nx);
        specfem::assembly::load_on_device(index, properties, device_points(i));
      });
  const auto host_points =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_points);

  std::size_t samples = 0;
  std::size_t tiso_samples = 0;
  std::array<std::size_t, 3> region_samples{};
  for (int e = 0; e < elements.extent(0); ++e) {
    const int ispec = elements(e);
    const auto region = types.get_region_tag(ispec);
    EXPECT_EQ(types.get_medium_tag(ispec), region == Region::outer_core
                                               ? Medium::acoustic
                                               : Medium::elastic);
    for (int p = 0; p < npoints; ++p) {
      const Index index(ispec, p / (nx * ny), (p / nx) % ny, p % nx);
      Point point;
      specfem::assembly::load_on_host(index, properties, point);
      ASSERT_GT(point.rho(), 0.0);
      ASSERT_GT(point.kappa(), 0.0);
      if constexpr (MediumTag == Medium::elastic) {
        ASSERT_GT(point.mu(), 0.0);
      } else {
        ASSERT_GT(point.rho_inverse(), 0.0);
      }
      for (int field = 0; field < Point::nprops; ++field) {
        ASSERT_NEAR(point[field], host_points(e * npoints + p)[field],
                    std::numeric_limits<type_real>::epsilon() *
                        std::max(1.0, std::abs(double(point[field]))));
      }
    }

    // Sample elements throughout each medium, including both core regions.
    if (e % std::max(1, elements.extent(0) / 100) != 0 &&
        e != elements.extent(0) - 1) {
      continue;
    }
    std::vector<double> xyz(3 * npoints);
    for (int p = 0; p < npoints; ++p) {
      for (int d = 0; d < 3; ++d) {
        xyz[3 * p + d] = assembly.mesh.h_reference_coord(
            ispec, p / (nx * ny), (p / nx) % ny, p % nx, d);
      }
    }
    const int region_code = specfem::globe::to_region_code(region);
    const auto expected = oracle.evaluate_element(
        region_code, types.idoubling(ispec), types.rmin(ispec),
        types.rmax(ispec), types.elem_in_crust(ispec),
        types.elem_in_mantle(ispec), xyz);
    for (int p = 0; p < npoints; ++p) {
      const Index index(ispec, p / (nx * ny), (p / nx) % ny, p % nx);
      Point point;
      specfem::assembly::load_on_host(index, properties, point);
      const double rho = expected.rho[p];
      const double vp = expected.vp_iso[p];
      const double vs = expected.vs_iso[p];
      EXPECT_NEAR(point.rho(), rho, tolerance * rho);
      if constexpr (PropertyTag == Property::anisotropic) {
        double expected_stiffness[21];
        if (expected.is_anisotropic) {
          std::copy_n(&expected.cij[21 * p], 21, expected_stiffness);
        } else {
          // Independent of the production path: angles come straight from the
          // final GLL coordinates (not the spherical cache), and the rotation
          // is globe's own rotate_tensor_Love_to_global.
          double x[3];
          for (int d = 0; d < 3; ++d) {
            x[d] = assembly.mesh.h_coord(ispec, p / (nx * ny), (p / nx) % ny,
                                         p % nx, d);
          }
          const double radius =
              std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
          const double theta = std::acos(x[2] / radius);
          const double phi = std::atan2(x[1], x[0]);
          globe_evaluator_rotate_tiso_to_cij(
              theta, phi, expected.rho[p], expected.vpv[p], expected.vph[p],
              expected.vsv[p], expected.vsh[p], expected.eta[p],
              expected_stiffness);
          ++tiso_samples;
        }
        double tensor_scale = 1.0;
        for (int c = 0; c < 21; ++c) {
          tensor_scale =
              std::max(tensor_scale, std::abs(expected_stiffness[c]));
        }
        for (int c = 0; c < 21; ++c) {
          if (expected.is_anisotropic) {
            // Full anisotropy must be stored unrotated.
            EXPECT_EQ(point[c], static_cast<type_real>(expected_stiffness[c]));
          }
          EXPECT_NEAR(point[c], expected_stiffness[c], tolerance * tensor_scale)
              << "element " << ispec << ", point " << p << ", component " << c;
        }
      } else {
        const double mu = rho * vs * vs;
        const double kappa = rho * vp * vp - 4.0 / 3.0 * mu;
        EXPECT_NEAR(point.kappa(), kappa, tolerance * kappa);
        if constexpr (MediumTag == Medium::elastic) {
          EXPECT_NEAR(point.mu(), mu, tolerance * mu);
        } else {
          EXPECT_NEAR(point.rho_inverse(), 1.0 / rho, tolerance / rho);
        }
      }
      ++samples;
      ++region_samples[region_code - 1];
      if (fixture.mesh.globe.model_config.model_name == "1D_isotropic_prem") {
        const double radius = std::sqrt(xyz[3 * p] * xyz[3 * p] +
                                        xyz[3 * p + 1] * xyz[3 * p + 1] +
                                        xyz[3 * p + 2] * xyz[3 * p + 2]);
        const double r_prem =
            std::clamp(radius, double(types.rmin(ispec)) * 1.000001,
                       double(types.rmax(ispec)) * 0.999999);
        const auto prem =
            oracle.prem_reference(r_prem, types.idoubling(ispec), region_code);
        EXPECT_NEAR(point.rho(), prem.rho, tolerance * prem.rho);
        EXPECT_NEAR(point.vp(), prem.vp_iso, tolerance * prem.vp_iso);
        EXPECT_NEAR(point.vs(), prem.vs_iso,
                    tolerance * std::max(1.0, prem.vs_iso));
        if (profile.is_open()) {
          profile << r_prem << ',' << point.rho() << ',' << point.vp() << ','
                  << point.vs() << ',' << prem.rho << ',' << prem.vp_iso << ','
                  << prem.vs_iso << '\n';
        }
      }
    }
  }
  EXPECT_GE(samples, 100u);
  if (PropertyTag == Property::anisotropic &&
      fixture.mesh.globe.model_config.model_name ==
          "1D_transversely_isotropic_prem") {
    // #2045 acceptance: at least 50 TISO GLL points against globe's rotation.
    EXPECT_GE(tiso_samples, 50u);
  }
  for (const auto region :
       { Region::crust_mantle, Region::outer_core, Region::inner_core }) {
    bool present = false;
    for (int e = 0; e < elements.extent(0); ++e) {
      present |= types.get_region_tag(elements(e)) == region;
    }
    if (present) {
      EXPECT_GE(region_samples[specfem::globe::to_region_code(region) - 1],
                100u);
    }
  }
}

class GlobeProperties : public ::testing::TestWithParam<bool> {};

class RecordingReader : public specfem::io::reader {
public:
  bool called = false;
  void read(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                &) override {}
  void read(Assembly &) override { called = true; }
};

} // namespace specfem::assembly::globe_properties_test_impl

using specfem::assembly::globe_properties_test_impl::GlobeProperties;

TEST_P(GlobeProperties, MatchesOracleOnHostAndDevice) {
  using specfem::assembly::globe_properties_test_impl::Medium;
  using specfem::assembly::globe_properties_test_impl::Property;
  const bool anisotropic = GetParam();
  specfem::assembly::globe_properties_test_impl::GlobeFixture fixture(
      anisotropic);
  ASSERT_TRUE(fixture.mesh.globe.has_reference_geometry);
  ASSERT_NE(fixture.assembly.mesh.h_reference_coord.data(),
            fixture.assembly.mesh.h_coord.data());
  specfem::assembly::dim3_impl::read_globe_properties(fixture.mesh,
                                                      fixture.assembly);
  std::ofstream profile;
  if (!anisotropic) {
    if (const char *path = std::getenv("SPECFEM_GLOBE_PROFILE")) {
      profile.open(path);
      ASSERT_TRUE(profile.is_open());
      profile << std::setprecision(17)
              << "radius,rho,vp,vs,prem_rho,prem_vp,prem_vs\n";
    }
  }
  {
    const specfem::globe::ModelEvaluator oracle(
        fixture.mesh.globe.model_config);
    specfem::assembly::globe_properties_test_impl::check_properties<
        Medium::elastic, Property::isotropic>(fixture, oracle, profile);
    specfem::assembly::globe_properties_test_impl::check_properties<
        Medium::acoustic, Property::isotropic>(fixture, oracle, profile);
    if (anisotropic) {
      specfem::assembly::globe_properties_test_impl::check_properties<
          Medium::elastic, Property::anisotropic>(fixture, oracle, profile);
    }
  }
  if (anisotropic) {
    fixture.mesh.globe.model_config.model_name =
        "1D_transversely_isotropic_prem_ACM";
    // Full anisotropy is already Cartesian and needs no spherical cache.
    fixture.assembly.mesh.h_spherical_coord = {};
    ASSERT_NO_THROW(specfem::assembly::dim3_impl::read_globe_properties(
        fixture.mesh, fixture.assembly));
    const specfem::globe::ModelEvaluator oracle(
        fixture.mesh.globe.model_config);
    specfem::assembly::globe_properties_test_impl::check_properties<
        Medium::elastic, Property::anisotropic>(fixture, oracle, profile);
  }
}

INSTANTIATE_TEST_SUITE_P(Globe, GlobeProperties, ::testing::Bool());

TEST(GlobeProperties, RejectsAzimuthalAnisotropy) {
  specfem::globe::ModelEvaluator::ElementProperties values;
  values.gc_prime.assign(125, 0.0);
  values.gs_prime.assign(125, 0.0);
  EXPECT_NO_THROW(
      specfem::assembly::dim3_impl::reject_azimuthal_anisotropy(values, 3));

  values.gc_prime[42] = 1.e-3;
  EXPECT_THROW(
      specfem::assembly::dim3_impl::reject_azimuthal_anisotropy(values, 3),
      std::runtime_error);

  values.gc_prime[42] = 0.0;
  values.gs_prime[124] = -2.e-4;
  try {
    specfem::assembly::dim3_impl::reject_azimuthal_anisotropy(values, 3);
    FAIL() << "non-zero Gs_prime was silently accepted";
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("compute element 3"), std::string::npos);
    EXPECT_NE(message.find("deferred"), std::string::npos);
  }

  // Full anisotropy: Gc'/Gs' are derived from the stored cij, nothing is lost.
  values.is_anisotropic = true;
  EXPECT_NO_THROW(
      specfem::assembly::dim3_impl::reject_azimuthal_anisotropy(values, 3));
}

TEST(GlobePropertyReader, BypassesOracleInitialization) {
  specfem::mesh::globe3d_mesh mesh;
  // No constants/context and an invalid model: entering the oracle path fails.
  mesh.globe.model_config.model_name = "invalid_model";
  specfem::assembly::globe_properties_test_impl::Assembly assembly;
  const auto reader = std::make_shared<
      specfem::assembly::globe_properties_test_impl::RecordingReader>();
  EXPECT_NO_THROW(specfem::assembly::dim3_impl::read_deferred_properties(
      mesh, assembly, reader));
  EXPECT_TRUE(reader->called);
  EXPECT_FALSE(specfem::globe::ModelEvaluator::is_active());
  EXPECT_THROW(specfem::assembly::dim3_impl::read_deferred_properties(
                   mesh, assembly, nullptr),
               std::runtime_error);
}

TEST(GlobeGravity, PrecomputesReferenceTablesAndDeviceData) {
  using specfem::assembly::globe_properties_test_impl::GlobeFixture;
  GlobeFixture fixture(false);
  auto &assembly = fixture.assembly;
  auto &globe = fixture.mesh.globe;
  using Medium = specfem::element::medium_tag;
  using Field = specfem::simulation::field_type;
  using Region = specfem::element::region_tag;

  // Combined runs have no forward field. Gravity must share the active field's
  // numbering regardless of simulation type.
  assembly.fields = { assembly.mesh, assembly.element_types,
                      specfem::simulation::type::combined };
  const auto reader = std::make_shared<
      specfem::assembly::globe_properties_test_impl::RecordingReader>();
  specfem::assembly::dim3_impl::read_deferred_properties(fixture.mesh, assembly,
                                                         reader);
  EXPECT_TRUE(assembly.gravity.radial.density.empty());
  EXPECT_EQ(assembly.gravity.gravity_pre_store.data(), nullptr);
  EXPECT_EQ(assembly.gravity.gravity_h.data(), nullptr);
  EXPECT_EQ(assembly.gravity.acoustic_gravity_pre_store.data(), nullptr);
  EXPECT_EQ(assembly.gravity.gravity_rho_g_over_kappa.data(), nullptr);

  globe.model_config.gravity = true;
  globe.full_gravity = true;
  const auto &types = assembly.element_types;
  const auto &field = assembly.fields.get_simulation_field<Field::adjoint>();
  const auto &coord = assembly.mesh.h_spherical_coord;
  const int nx = assembly.mesh.element_grid.ngllx;
  const int ny = assembly.mesh.element_grid.nglly;
  const int nz = assembly.mesh.element_grid.ngllz;

  // Approach the centre along two axes and straddle the 100 m handoff.
  // Update every element-local copy of each probe before setup.
  constexpr std::array<std::array<double, 3>, 6> positions{ { { 0, 0, 0 },
                                                              { 25, 0, 0 },
                                                              { 50, 0, 0 },
                                                              { 0, -50, 0 },
                                                              { 99.99, 0, 0 },
                                                              { 100, 0, 0 } } };
  ASSERT_GE(nx * ny, positions.size());
  std::array<int, positions.size()> probes;
  probes.fill(-1);
  const auto &xyz = assembly.mesh.h_coord;
  for (int ispec = 0; ispec < assembly.mesh.nspec; ++ispec) {
    if (types.get_region_tag(ispec) != Region::inner_core) {
      continue;
    }
    if (probes[0] < 0) {
      for (int i = 0; i < static_cast<int>(probes.size()); ++i) {
        probes[i] =
            field.get_iglob<false, Medium::elastic>(ispec, 0, i / nx, i % nx);
      }
    }
    for (int iz = 0; iz < nz; ++iz) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          const int point =
              field.get_iglob<false, Medium::elastic>(ispec, iz, iy, ix);
          for (std::size_t i = 0; i < probes.size(); ++i) {
            if (point == probes[i]) {
              for (int c = 0; c < 3; ++c) {
                xyz(ispec, iz, iy, ix, c) = positions[i][c];
              }
            }
          }
        }
      }
    }
  }
  ASSERT_GE(probes.front(), 0);
  assembly.mesh.compute_spherical_coordinates();

  specfem::assembly::dim3_impl::read_deferred_properties(fixture.mesh, assembly,
                                                         reader);
  const auto &gravity = assembly.gravity;
  const auto &radial = gravity.radial;
  ASSERT_EQ(radial.minus_gravity.size(), 70000);
  ASSERT_EQ(radial.minus_deriv_gravity.size(), 70000);
  ASSERT_EQ(radial.density.size(), 70000);
  ASSERT_EQ(radial.d_ln_density_dr.size(), 70000);
  ASSERT_EQ(radial.minus_rho_g_over_kappa_fluid.size(), 70000);
  const double planet_radius = globe.planet_constants->r_planet();
  const double step = planet_radius * radial.range_max / 70000;
  EXPECT_NEAR(step * 70000,
              planet_radius + globe.planet_constants->topo_maximum(), 1.e-8);
  EXPECT_NEAR(-radial.minus_gravity[std::lround(planet_radius / step) - 1],
              9.81, 0.1);
  for (std::size_t i = 0; i < radial.density.size(); ++i) {
    ASSERT_GT(radial.density[i], 0.0) << i;
    ASSERT_TRUE(std::isfinite(radial.minus_deriv_gravity[i])) << i;
    ASSERT_TRUE(std::isfinite(radial.minus_rho_g_over_kappa_fluid[i])) << i;
    ASSERT_NEAR(radial.d_ln_density_dr[i], 0.0, 0.0) << i;
    if (i > 0 && (i + 1) * step < 3400000.0) {
      ASSERT_LT(radial.minus_gravity[i], radial.minus_gravity[i - 1]) << i;
    }
  }

  const auto solid = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), gravity.gravity_pre_store);
  const auto tensor = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          gravity.gravity_h);
  const auto fluid = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), gravity.acoustic_gravity_pre_store);
  const auto factor = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), gravity.gravity_rho_g_over_kappa);
  ASSERT_EQ(solid.extent(0), field.get_nglob<Medium::elastic>());
  ASSERT_EQ(tensor.extent(0), solid.extent(0));
  ASSERT_EQ(fluid.extent(0), field.get_nglob<Medium::acoustic>());
  ASSERT_EQ(factor.extent(0), fluid.extent(0));
  const double center_diagonal = radial.center_gradient * radial.center_density;
  ASSERT_LT(center_diagonal, 0.0);
  const double rel_tol = 2.0 * std::numeric_limits<type_real>::epsilon();
  for (std::size_t i = 0; i + 1 < probes.size(); ++i) {
    for (int c = 0; c < 3; ++c) {
      const double expected = center_diagonal * positions[i][c];
      EXPECT_NEAR(solid(probes[i], c), expected, rel_tol * std::abs(expected));
      EXPECT_NEAR(tensor(probes[i], c), center_diagonal,
                  rel_tol * std::abs(center_diagonal));
      EXPECT_NEAR(tensor(probes[i], c + 3), 0.0, 0.0);
    }
  }
  // At 100 m PREM selects its first radial sample (about 91 m).
  const int boundary = probes.back();
  const double weight = radial.density.front() * radial.minus_gravity.front();
  EXPECT_NEAR(solid(boundary, 0), weight, rel_tol * std::abs(weight));
  EXPECT_NEAR(solid(boundary, 1), 0.0, 0.0);
  EXPECT_NEAR(solid(boundary, 2), 0.0, 0.0);
  EXPECT_NEAR(tensor(boundary, 1), weight / 100.0,
              rel_tol * std::abs(weight / 100.0));
  for (std::size_t point = 0; point < factor.extent(0); ++point) {
    const double norm =
        std::hypot(fluid(point, 0), fluid(point, 1), fluid(point, 2));
    ASSERT_LT(factor(point), 0.0);
    ASSERT_NEAR(norm, -factor(point), 1.e-12) << point;
  }

  if (const char *path = std::getenv("SPECFEM_GLOBE_GRAVITY_PROFILE")) {
    std::ofstream profile(path);
    ASSERT_TRUE(profile.is_open());
    profile << std::setprecision(17)
            << "r_m,minus_g,minus_dg,density,d_ln_density_dr,minus_rho_g_over_"
               "kappa\n";
    for (std::size_t i = 0; i < radial.density.size(); ++i) {
      profile << (i + 1) * step << ',' << radial.minus_gravity[i] << ','
              << radial.minus_deriv_gravity[i] << ',' << radial.density[i]
              << ',' << radial.d_ln_density_dr[i] << ','
              << radial.minus_rho_g_over_kappa_fluid[i] << '\n';
    }

    std::ofstream points(std::string(path) + ".points.csv");
    ASSERT_TRUE(points.is_open());
    points << std::setprecision(17)
           << "region,r,theta,phi,gx,gy,gz,hxx,hyy,hzz,hxy,hxz,hyz\n";
    std::array<int, 3> count{};
    std::vector<bool> solid_sampled(solid.extent(0));
    std::vector<bool> fluid_sampled(fluid.extent(0));
    for (int ispec = 0; ispec < assembly.mesh.nspec; ++ispec) {
      const int region =
          specfem::globe::to_region_code(types.get_region_tag(ispec));
      if (region < 1 || region > 3 || count[region - 1] >= 60) {
        continue;
      }
      const int iz = ispec % nz;
      const int iy = ispec % ny;
      const int ix = ispec % nx;
      const double radius = coord(ispec, iz, iy, ix, 0);
      const bool acoustic = types.get_medium_tag(ispec) == Medium::acoustic;
      const int point =
          acoustic ? field.get_iglob<false, Medium::acoustic>(ispec, iz, iy, ix)
                   : field.get_iglob<false, Medium::elastic>(ispec, iz, iy, ix);
      auto &sampled = acoustic ? fluid_sampled : solid_sampled;
      if (radius < 100.0 || sampled[point]) {
        continue;
      }
      sampled[point] = true;
      ++count[region - 1];
      points << region << ',' << radius << ',' << coord(ispec, iz, iy, ix, 1)
             << ',' << coord(ispec, iz, iy, ix, 2);
      if (acoustic) {
        for (int c = 0; c < 3; ++c) {
          points << ',' << fluid(point, c);
        }
        points << ",0,0,0,0,0,0\n";
      } else {
        for (int c = 0; c < 3; ++c) {
          points << ',' << solid(point, c);
        }
        for (int c = 0; c < 6; ++c) {
          points << ',' << tensor(point, c);
        }
        points << '\n';
      }
    }
    for (int sampled : count) {
      EXPECT_GE(sampled, 30);
    }
    EXPECT_GE(count[0] + count[1] + count[2], 100);
  }
}

TEST(GlobeGravity, AssemblyReleasesSphericalCache) {
  using specfem::assembly::globe_properties_test_impl::Assembly;
  constexpr auto dim = Assembly::dimension_tag;
  auto mesh = specfem::io::read_globe_mesh(
      "data/dim3_globe/GlobalSmallMesh/DATABASES_MPI/"
      "proc000000_specfempp_database.bin",
      specfem::attenuation::Setup{});
  mesh.globe.model_config.gravity = true;
  const specfem::quadrature::quadratures quadrature(
      specfem::quadrature::gll::gll{});
  std::vector<std::shared_ptr<specfem::sources::source<dim>>> sources;
  const Assembly assembly(mesh, quadrature, sources, {}, {}, 0.0, 0.1, 1, 1, 1,
                          specfem::simulation::type::forward, false, nullptr);
  EXPECT_EQ(assembly.mesh.h_spherical_coord.data(), nullptr);
  EXPECT_EQ(assembly.gravity.radial.minus_gravity.size(), 70000);
  EXPECT_GT(assembly.gravity.gravity_pre_store.extent(0), 0);
  EXPECT_GT(assembly.gravity.acoustic_gravity_pre_store.extent(0), 0);
  EXPECT_EQ(assembly.gravity.gravity_rho_g_over_kappa.data(), nullptr);
}

TEST(GlobeGravity, CartesianAssemblyLeavesGravityEmpty) {
  using specfem::assembly::globe_properties_test_impl::Assembly;
  constexpr auto dim = Assembly::dimension_tag;
  const auto mesh = specfem::io::read_3d_mesh(
      "data/dim3/EightNodeElastic/database.bin", specfem::attenuation::Setup{});
  const specfem::quadrature::quadratures quadrature(
      specfem::quadrature::gll::gll{});
  std::vector<std::shared_ptr<specfem::sources::source<dim>>> sources;
  const Assembly assembly(mesh, quadrature, sources, {}, {}, 0.0, 0.1, 1, 1, 1,
                          specfem::simulation::type::forward, false, nullptr);
  EXPECT_TRUE(assembly.gravity.radial.density.empty());
  EXPECT_EQ(assembly.gravity.gravity_pre_store.data(), nullptr);
  EXPECT_EQ(assembly.gravity.gravity_h.data(), nullptr);
  EXPECT_EQ(assembly.gravity.acoustic_gravity_pre_store.data(), nullptr);
  EXPECT_EQ(assembly.gravity.gravity_rho_g_over_kappa.data(), nullptr);
}
