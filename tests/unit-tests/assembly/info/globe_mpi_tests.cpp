#include "../dim3/test_fixture.hpp"
#include "../dim3_globe/test_fixture.hpp"
#include "SPECFEM_Environment.hpp"
#include "specfem/assembly/info/impl/globe.hpp"
#include "specfem/mpi.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>
#include <yaml-cpp/yaml.h>

TEST(GlobeInfo, LocationReductionDistinguishesAdjacentValues) {
  const int rank = specfem::MPI::get_rank();
  // Simulate a last-bit difference between timestep calculations. Rank 1
  // owns the smaller value even though rank 0 wins an exact tie.
  const type_real value =
      rank == 1 ? type_real(1) : std::nextafter(type_real(1), type_real(2));
  specfem::assembly::InfoLocation location{
    rank, rank + 2, rank + 3, specfem::element::region_tag::outer_core,
    1000.0 + rank
  };
  specfem::assembly::info_impl::reduce_location(value, location);
  EXPECT_EQ(location.rank, 1);
  EXPECT_EQ(location.compute_element, 3);
  EXPECT_EQ(location.mesh_element, 4);
  EXPECT_EQ(location.region, specfem::element::region_tag::outer_core);
  EXPECT_EQ(location.radius, 1001.0);
}

TEST(GlobeInfo, LocationReductionBreaksTiesAndIgnoresEmptyRanks) {
  const int rank = specfem::MPI::get_rank();
  specfem::assembly::InfoLocation location{
    rank, rank == 0 ? -1 : 2, 3, specfem::element::region_tag::inner_core,
    1000.0 + rank
  };
  const type_real value =
      rank == 0 ? std::numeric_limits<type_real>::max() : type_real(1);
  specfem::assembly::info_impl::reduce_location(value, location);
  EXPECT_EQ(location.rank, 1);
  EXPECT_EQ(location.compute_element, 2);
  EXPECT_EQ(location.mesh_element, 3);
  EXPECT_EQ(location.region, specfem::element::region_tag::inner_core);
  EXPECT_EQ(location.radius, 1001.0);
}

TEST(GlobeInfo, LocationReductionResetsWhenNoRankOwnsAnElement) {
  // Seed different stale diagnostics on each rank to ensure the no-owner
  // path restores every field to the same default value.
  const int rank = specfem::MPI::get_rank();
  specfem::assembly::InfoLocation location{
    rank, -1, rank, specfem::element::region_tag::inner_core, 1000.0 + rank
  };
  specfem::assembly::info_impl::reduce_location(
      std::numeric_limits<type_real>::max(), location);
  const specfem::assembly::InfoLocation empty;
  EXPECT_EQ(location.rank, empty.rank);
  EXPECT_EQ(location.compute_element, empty.compute_element);
  EXPECT_EQ(location.mesh_element, empty.mesh_element);
  EXPECT_EQ(location.region, empty.region);
  EXPECT_EQ(location.radius, empty.radius);
}

TEST(GlobeInfo, RejectsMixedContextOnEveryRank) {
  constexpr auto dimension = specfem::element::dimension_tag::dim3;
  specfem::test_configuration::Assembly3D fixture(
      "data/dim3/EightNodeElastic/database.bin");
  auto &assembled = fixture.assembly;
  if (specfem::MPI::get_rank() == 1) {
    assembled.element_types.regions = decltype(assembled.element_types.regions)(
        "mixed_context", assembled.mesh.nspec);
    Kokkos::deep_copy(assembled.element_types.regions,
                      specfem::element::region_tag::crust_mantle);
  }
  EXPECT_THROW((specfem::assembly::Info<dimension>(
                   assembled.mesh, assembled.properties,
                   assembled.element_types, &assembled.jacobian_matrix)),
               std::runtime_error);
}

TEST(GlobeInfo, SixChunkMesherCrossCheck) {
  ASSERT_TRUE(SPECFEMEnvironment::IsMPISizeValid());
  const std::string fixture = "data/dim3_globe/GlobalSphereMesh/";
  std::ostringstream filename;
  filename << fixture << "DATABASES_MPI/proc" << std::setfill('0')
           << std::setw(6) << specfem::MPI::get_rank()
           << "_specfempp_database.bin";
  const specfem::test_configuration::GlobeAssembly3D globe(filename.str());
  const auto &info = globe.assembly.info;
  const auto &constants = *globe.mesh.globe.planet_constants;
  const auto reference = YAML::LoadFile(fixture + "provenance/mesher.yaml");
  const double radius = constants.r_planet();
  // Fixed planet constants are separate from model discontinuity radii.
  // PREM_RCMB and PREM_RICB are recorded in this fixture's provenance.
  const double r_cmb = reference["r_cmb_m"].as<double>();
  const double r_icb = reference["r_icb_m"].as<double>();
  const double sphere_volume =
      4.0 / 3.0 * std::acos(-1.0) * std::pow(radius, 3);
  const double mesher_volume =
      reference["volume_unit_sphere"].as<double>() * std::pow(radius, 3);
  const double mesher_mass = reference["mass_kg"].as<double>();
  const double dt = reference["dt_seconds"].as<double>();
  EXPECT_NEAR(info.total_volume, sphere_volume, 0.001 * sphere_volume);
  // Mesher uses single-precision Jacobians; C++ recomputes geometry in the
  // configured precision. Allow quadrature/roundoff differences in both builds.
  EXPECT_NEAR(info.total_volume, mesher_volume, 1.e-5 * mesher_volume);
  EXPECT_NEAR(info.total_mass, mesher_mass, 0.001 * mesher_mass);
  EXPECT_NEAR(info.total_mass, 5.972e24, 0.001 * 5.972e24);
  EXPECT_GE(info.suggested_time_step, dt / 2);
  EXPECT_LE(info.suggested_time_step, dt * 2);

  using Region = specfem::element::region_tag;
  ASSERT_EQ(info.regions.size(), 3);
  const auto &mantle = info.regions.at(Region::crust_mantle);
  const auto &outer = info.regions.at(Region::outer_core);
  const auto &inner = info.regions.at(Region::inner_core);
  // The coarse 27-node mapping approximates a sphere between its anchors.
  // Allow 0.2% interpolation error for physical GLL radii; independently
  // require the mesher's shell context to agree with PREM's reference radii to
  // 1 m.
  constexpr double radial_interpolation_tolerance = 0.002;
  EXPECT_NEAR(mantle.radius.min, r_cmb, radial_interpolation_tolerance * r_cmb);
  EXPECT_NEAR(mantle.radius.max, radius,
              radial_interpolation_tolerance * radius);
  EXPECT_NEAR(outer.radius.min, r_icb, radial_interpolation_tolerance * r_icb);
  EXPECT_NEAR(outer.radius.max, r_cmb, radial_interpolation_tolerance * r_cmb);
  EXPECT_NEAR(inner.radius.min, 0, 1);
  EXPECT_NEAR(inner.radius.max, r_icb, radial_interpolation_tolerance * r_icb);
  const auto comm = specfem::MPI::communicator();
  for (const auto region :
       { Region::crust_mantle, Region::outer_core, Region::inner_core }) {
    double lower = std::numeric_limits<double>::max(), upper = 0;
    const auto &types = globe.assembly.element_types;
    for (int ispec = 0; ispec < globe.assembly.mesh.nspec; ++ispec) {
      if (types.get_region_tag(ispec) == region) {
        lower = std::min(lower, static_cast<double>(types.rmin(ispec)));
        upper = std::max(upper, static_cast<double>(types.rmax(ispec)));
      }
    }
    SPECFEM_MPI_SAFECALL(
        MPI_Allreduce(MPI_IN_PLACE, &lower, 1, MPI_DOUBLE, MPI_MIN, comm));
    SPECFEM_MPI_SAFECALL(
        MPI_Allreduce(MPI_IN_PLACE, &upper, 1, MPI_DOUBLE, MPI_MAX, comm));
    const double expected_lower = region == Region::crust_mantle ? r_cmb
                                  : region == Region::outer_core ? r_icb
                                                                 : 0;
    const double expected_upper = region == Region::crust_mantle ? radius
                                  : region == Region::outer_core ? r_cmb
                                                                 : r_icb;
    EXPECT_NEAR(lower, expected_lower, 1);
    EXPECT_NEAR(upper, expected_upper, 1);
  }

  std::vector<double> values{ info.total_volume, info.total_mass,
                              info.suggested_time_step, info.gll_distance.min };
  int count = 0;
  for (const auto &[region, statistics] : info.regions) {
    EXPECT_GT(statistics.element_count, 0);
    EXPECT_EQ(statistics.element_count, info.elements_per_region.at(region));
    EXPECT_GT(statistics.minimum_gll_spacing, 0);
    EXPECT_GT(statistics.wave_speed.min, 0);
    EXPECT_LT(statistics.wave_speed.max, 16000);
    count += statistics.element_count;
    values.insert(values.end(),
                  { static_cast<double>(statistics.element_count),
                    statistics.radius.min, statistics.radius.max,
                    statistics.wave_speed.min, statistics.wave_speed.max,
                    statistics.minimum_gll_spacing,
                    statistics.suggested_time_step });
  }
  int global_element_count = globe.assembly.mesh.nspec;
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &global_element_count, 1,
                                     MPI_INT, MPI_SUM, comm));
  EXPECT_EQ(count, global_element_count);
  for (const auto &location :
       { info.minimum_spacing_location, info.cfl_location }) {
    EXPECT_GE(location.rank, 0);
    EXPECT_LT(location.rank, 6);
    EXPECT_GE(location.compute_element, 0);
    EXPECT_GE(location.mesh_element, 0);
    EXPECT_GE(location.radius, info.regions.at(location.region).radius.min);
    EXPECT_LE(location.radius, info.regions.at(location.region).radius.max);
    values.insert(values.end(),
                  { static_cast<double>(location.rank),
                    static_cast<double>(location.compute_element),
                    static_cast<double>(location.mesh_element),
                    static_cast<double>(location.region), location.radius });
  }
  auto minima = values, maxima = values;
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, minima.data(), minima.size(),
                                     MPI_DOUBLE, MPI_MIN, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, maxima.data(), maxima.size(),
                                     MPI_DOUBLE, MPI_MAX, comm));
  EXPECT_EQ(minima, maxima); // Every reported value is identical on all ranks.
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(6));
  return RUN_ALL_TESTS();
}
