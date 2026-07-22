#include "../../SPECFEM_Environment.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
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
template <specfem::element::dimension_tag DimensionTag> struct SourceTestParam {
  std::string testname;
  std::string sourcefilename;
  specfem::enums::source_format format;
  std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
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
template <specfem::element::dimension_tag DimensionTag>
std::ostream &operator<<(std::ostream &os,
                         const SourceTestParam<DimensionTag> &params) {
  os << params.testname;
  return os;
}

using SourceTestParam2D =
    SourceTestParam<specfem::element::dimension_tag::dim2>;

class Read2DSourcesTest : public ::testing::TestWithParam<SourceTestParam2D> {};

TEST_P(Read2DSourcesTest, ReadSources) {
  const auto &param = GetParam();

  std::vector<specfem::enums::source_file_entry> entries = {
    { param.format, param.sourcefilename }
  };

  auto [sources, _t0, _starttime] =
      specfem::io::read_sources<specfem::element::dimension_tag::dim2>(
          entries, nsteps, user_t0, dt, specfem::simulation::type::forward);

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
                           specfem::enums::source_format::YAML,
                           single_moment_tensor_2d },
        SourceTestParam2D{
            "2D Single Force", "io/sources/data/dim2/single_force.yaml",
            specfem::enums::source_format::YAML, single_force_2d },
        SourceTestParam2D{ "2D Single Cosserat Force",
                           "io/sources/data/dim2/single_cosserat_force.yaml",
                           specfem::enums::source_format::YAML,
                           single_cosserat_force_2d },
        SourceTestParam2D{
            "2D Multiple Sources", "io/sources/data/dim2/multiple_sources.yaml",
            specfem::enums::source_format::YAML, multiple_sources_2d }));

using SourceTestParam3D =
    SourceTestParam<specfem::element::dimension_tag::dim3>;

class Read3DSourcesTest : public ::testing::TestWithParam<SourceTestParam3D> {};

TEST_P(Read3DSourcesTest, ReadSources) {
  const auto &param = GetParam();

  std::vector<specfem::enums::source_file_entry> entries = {
    { param.format, param.sourcefilename }
  };

  auto [sources, _t0, _starttime] =
      specfem::io::read_sources<specfem::element::dimension_tag::dim3>(
          entries, nsteps, user_t0, dt, specfem::simulation::type::forward);

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
        SourceTestParam3D{
            "3D Single Force", "io/sources/data/dim3/single_force.yaml",
            specfem::enums::source_format::YAML, single_force_3d },
        SourceTestParam3D{ "3D Single Moment Tensor",
                           "io/sources/data/dim3/single_moment_tensor.yaml",
                           specfem::enums::source_format::YAML,
                           single_moment_tensor_3d },
        SourceTestParam3D{
            "3D Multiple Sources", "io/sources/data/dim3/multiple_sources.yaml",
            specfem::enums::source_format::YAML, multiple_sources_3d },
        SourceTestParam3D{ "3D Single Cosserat Force",
                           "io/sources/data/dim3/single_cosserat_force.yaml",
                           specfem::enums::source_format::YAML,
                           single_cosserat_force_3d },
        SourceTestParam3D{
            "3D CMTSOLUTION Single Moment Tensor",
            "io/sources/data/dim3/single_moment_tensor.CMTSOLUTION",
            specfem::enums::source_format::CMTSOLUTION,
            single_moment_tensor_cmt_3d },
        SourceTestParam3D{
            "3D CMTSOLUTION Spherical Moment Tensor",
            "io/sources/data/dim3/spherical_moment_tensor.CMTSOLUTION",
            specfem::enums::source_format::CMTSOLUTION,
            spherical_moment_tensor_cmt_3d },
        SourceTestParam3D{
            "3D CMTSOLUTION Geographic Moment Tensor",
            "io/sources/data/dim3/single_moment_tensor_geographic.CMTSOLUTION",
            specfem::enums::source_format::CMTSOLUTION,
            single_moment_tensor_geographic_cmt_3d },
        SourceTestParam3D{ "3D FORCESOLUTION Single Force",
                           "io/sources/data/dim3/single_force.FORCESOLUTION",
                           specfem::enums::source_format::FORCESOLUTION,
                           single_force_forcesolution_3d },
        SourceTestParam3D{
            "3D FORCESOLUTION Geographic Single Force",
            "io/sources/data/dim3/single_force_geographic.FORCESOLUTION",
            specfem::enums::source_format::FORCESOLUTION,
            single_force_geographic_forcesolution_3d },
        SourceTestParam3D{ "3D CMTSOLUTION Multiple Sources",
                           "io/sources/data/dim3/multiple_sources.CMTSOLUTION",
                           specfem::enums::source_format::CMTSOLUTION,
                           multiple_sources_cmt_3d },
        SourceTestParam3D{ "3D FORCESOLUTION Multiple Forces",
                           "io/sources/data/dim3/multiple_forces.FORCESOLUTION",
                           specfem::enums::source_format::FORCESOLUTION,
                           multiple_forces_forcesolution_3d }));
