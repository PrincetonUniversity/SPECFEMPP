#include "specfem/assembly/assembly/dim3/globe_properties.hpp"
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
      mesh.globe.model_config.model_name = "1D_transversely_isotropic_prem_ACM";
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
        for (int c = 0; c < 21; ++c) {
          EXPECT_NEAR(point[c], expected.cij[21 * p + c],
                      tolerance *
                          std::max(1.0, std::abs(expected.cij[21 * p + c])));
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
    // An isotropic oracle must never silently populate anisotropic storage.
    fixture.mesh.globe.model_config.model_name = "1D_isotropic_prem";
    EXPECT_THROW(specfem::assembly::dim3_impl::read_globe_properties(
                     fixture.mesh, fixture.assembly),
                 std::runtime_error);
  }
}

INSTANTIATE_TEST_SUITE_P(Globe, GlobeProperties, ::testing::Bool());

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
