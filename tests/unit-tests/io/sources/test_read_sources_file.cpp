#include "../../Kokkos_Environment.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "io/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "test_source_solutions.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <gtest/gtest.h>

// Local constants since these would be set by the simulation.
extern int nsteps;
extern type_real dt;
extern int tshift;
extern type_real user_t0;

/**
 * @brief Parameters for testing source reading.
 *
 * @tparam DimensionTag
 */
template <specfem::dimension::type DimensionTag> struct SourceTestParam {
  std::string testname;
  std::string sourcefilename;
  std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
      expected_sources;
};

/**
 * @brief Stream insertion operator for SourceTestParam.
 *
 * @tparam DimensionTag
 * @param os
 * @param params
 * @return std::ostream&
 */
template <specfem::dimension::type DimensionTag>
std::ostream &operator<<(std::ostream &os,
                         const SourceTestParam<DimensionTag> &params) {
  os << params.testname;
  return os;
}

using SourceTestParam2D = SourceTestParam<specfem::dimension::type::dim2>;

class Read2DSourcesTest : public ::testing::TestWithParam<SourceTestParam2D> {};

TEST_P(Read2DSourcesTest, ReadSources) {
  const auto &param = GetParam();

  auto [sources, _t0] =
      specfem::io::read_2d_sources(param.sourcefilename, nsteps, user_t0, dt,
                                   specfem::simulation::type::forward);

  ASSERT_EQ(sources.size(), param.expected_sources.size());

  for (size_t i = 0; i < sources.size(); ++i) {
    auto source = sources[i];
    auto expected_source = param.expected_sources[i];

    std::cout << "Act. Source type: " << typeid(source).name() << "\n";
    std::cout << "Exp. Source type: " << typeid(expected_source).name() << "\n";

    EXPECT_EQ(*source, *expected_source)
        << "Source mismatch at index " << i << ":\n"
        << "Expected:\n"
        << expected_source->print()
        << "\n"
           "!=\n"
        << "Actual:\n"
        << source->print() << "\n";
  }
}

INSTANTIATE_TEST_SUITE_P(
    IO_TESTS, Read2DSourcesTest,
    ::testing::Values(
        SourceTestParam2D{ "2D Single Moment Tensor",
                           "io/sources/data/dim2/single_moment_tensor.yaml",
                           single_moment_tensor_2d },
        SourceTestParam2D{ "2D Single Force",
                           "io/sources/data/dim2/single_force.yaml",
                           single_force_2d },
        SourceTestParam2D{ "2D Single Cosserat Force",
                           "io/sources/data/dim2/single_cosserat_force.yaml",
                           single_cosserat_force_2d },
        SourceTestParam2D{ "2D Multiple Sources",
                           "io/sources/data/dim2/multiple_sources.yaml",
                           multiple_sources_2d }));

using SourceTestParam3D = SourceTestParam<specfem::dimension::type::dim3>;

class Read3DSourcesTest : public ::testing::TestWithParam<SourceTestParam3D> {};

TEST_P(Read3DSourcesTest, ReadSources) {
  const auto &param = GetParam();

  auto [sources, _t0] =
      specfem::io::read_3d_sources(param.sourcefilename, nsteps, user_t0, dt,
                                   specfem::simulation::type::forward);

  ASSERT_EQ(sources.size(), param.expected_sources.size());

  for (size_t i = 0; i < sources.size(); ++i) {

    auto source = sources[i];
    auto expected_source = param.expected_sources[i];

    std::cout << "Act. Source type: " << typeid(source).name() << "\n";
    std::cout << "Exp. Source type: " << typeid(expected_source).name() << "\n";

    EXPECT_EQ(*source, *expected_source)
        << "Source mismatch at index " << i << ":\n"
        << "Expected:\n"
        << expected_source->print()
        << "\n"
           "!=\n"
        << "Actual:\n"
        << source->print() << "\n";
  }
}

INSTANTIATE_TEST_SUITE_P(
    IO_TESTS, Read3DSourcesTest,
    ::testing::Values(
        SourceTestParam3D{ "3D Single Force",
                           "io/sources/data/dim3/single_force.yaml",
                           single_force_3d },
        SourceTestParam3D{ "3D Single Moment Tensor",
                           "io/sources/data/dim3/single_moment_tensor.yaml",
                           single_moment_tensor_3d },
        SourceTestParam3D{ "3D Multiple Sources",
                           "io/sources/data/dim3/multiple_sources.yaml",
                           multiple_sources_3d }));
