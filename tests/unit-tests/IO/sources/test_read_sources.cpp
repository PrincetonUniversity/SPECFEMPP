#include "../../Kokkos_Environment.hpp"
#include "IO/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "source/interface.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

TEST(IO_TESTS, read_sources) {
  /**
   *  This test checks whether a moment tensor source is read correctly
   */

  std::string sources_file = "IO/sources/data/single_moment_tensor.yml";

  YAML::Node databases;
  databases["sources"] = sources_file;

  int nsteps = 100;
  type_real t0 = 0.0;
  type_real dt = 0.01;
  type_real f0 = 1.0;
  type_real tshift = 30.0;
  type_real factor = 1.0e10;
  // This is from the definition of the STF such that it is zero at t=0
  type_real hdur = 1.0 / f0;
  type_real t0_final = -1.2 * hdur + tshift;

  std::cout << "hello" << std::endl;
  std::cout << databases["sources"].as<std::string>() << std::endl;
  auto [sources, user_t0] = specfem::IO::read_sources(
      databases["sources"], nsteps, t0, dt, specfem::simulation::type::forward);

  std::cout << "world" << std::endl;

  ASSERT_EQ(sources.size(), 1);
  auto source = sources[0];
  auto mt = std::dynamic_pointer_cast<specfem::sources::moment_tensor>(source);
  ASSERT_EQ(mt->get_wavefield_type(),
            specfem::wavefield::simulation_field::forward);
  ASSERT_EQ(mt->get_Mxx(), 1.0);
  ASSERT_EQ(mt->get_Mxz(), 0.0);
  ASSERT_EQ(mt->get_Mzz(), 1.0);
  EXPECT_NEAR(mt->get_x(), 2000.0, 1e-10);
  EXPECT_NEAR(mt->get_z(), 3000.0, 1e-10);
  ASSERT_EQ(source->get_t0(), t0_final);
  ASSERT_EQ(source->get_tshift(), 0.0); // tshift is adjusted to be 0.0
}

// int main(int argc, char *argv[]) {
//   ::testing::InitGoogleTest(&argc, argv);
//   ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
//   return RUN_ALL_TESTS();
// }
