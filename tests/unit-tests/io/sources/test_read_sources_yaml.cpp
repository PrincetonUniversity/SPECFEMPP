#include "../../SPECFEM_Environment.hpp"
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
#include <yaml-cpp/yaml.h>

// Local constants since these would be set by the simulation.
extern int nsteps;
extern type_real dt;
extern int tshift;
extern type_real user_t0;

/**
 * @brief Parameters for testing source reading from YAML nodes.
 *
 * @tparam DimensionTag
 */
template <specfem::dimension::type DimensionTag> struct SourceYAMLTestParam {
  std::string testname;
  YAML::Node sources_node;
  std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
      expected_sources;
};

/**
 * @brief Stream insertion operator for SourceYAMLTestParam.
 *
 * @tparam DimensionTag
 * @param os
 * @param params
 * @return std::ostream&
 */
template <specfem::dimension::type DimensionTag>
std::ostream &operator<<(std::ostream &os,
                         const SourceYAMLTestParam<DimensionTag> &params) {
  os << params.testname;
  return os;
}

using SourceYAMLTestParam2D =
    SourceYAMLTestParam<specfem::dimension::type::dim2>;
using SourceYAMLTestParam3D =
    SourceYAMLTestParam<specfem::dimension::type::dim3>;

// YAML node test data for 2D sources
const static YAML::Node single_moment_tensor_yaml_2d = []() {
  YAML::Node node;
  node["number-of-sources"] = 1;
  YAML::Node source;
  YAML::Node moment_tensor;
  moment_tensor["x"] = 2000.0;
  moment_tensor["z"] = 3000.0;
  moment_tensor["Mxx"] = 1.0;
  moment_tensor["Mzz"] = 1.0;
  moment_tensor["Mxz"] = 0.0;
  moment_tensor["Ricker"]["factor"] = 1.0e10;
  moment_tensor["Ricker"]["tshift"] = 30.0;
  moment_tensor["Ricker"]["f0"] = 1.0;
  source["moment-tensor"] = moment_tensor;
  node["sources"].push_back(source);
  return node;
}();

const static YAML::Node single_force_yaml_2d = []() {
  YAML::Node node;
  node["number-of-sources"] = 1;
  YAML::Node source;
  YAML::Node force;
  force["x"] = 2500.0;
  force["z"] = 2500.0;
  force["source_surf"] = false;
  force["angle"] = 0.0;
  force["vx"] = 0.0;
  force["vz"] = 0.0;
  force["Ricker"]["factor"] = 1e10;
  force["Ricker"]["tshift"] = 5.0;
  force["Ricker"]["f0"] = 10.0;
  source["force"] = force;
  node["sources"].push_back(source);
  return node;
}();

const static YAML::Node single_cosserat_force_yaml_2d = []() {
  YAML::Node node;
  node["number-of-sources"] = 1;
  YAML::Node source;
  YAML::Node cosserat_force;
  cosserat_force["x"] = 2500.0;
  cosserat_force["z"] = 2500.0;
  cosserat_force["source_surf"] = false;
  cosserat_force["angle"] = 0.0;
  cosserat_force["vx"] = 0.0;
  cosserat_force["vz"] = 0.0;
  cosserat_force["f"] = 0.0;
  cosserat_force["fc"] = 1.0;
  cosserat_force["Ricker"]["factor"] = 1.0e10;
  cosserat_force["Ricker"]["tshift"] = 0.0;
  cosserat_force["Ricker"]["f0"] = 10.0;
  source["cosserat-force"] = cosserat_force;
  node["sources"].push_back(source);
  return node;
}();

// YAML node test data for 3D sources
const static YAML::Node single_force_yaml_3d = []() {
  YAML::Node node;
  node["number-of-sources"] = 1;
  YAML::Node source;
  YAML::Node force;
  force["x"] = 2500.0;
  force["y"] = 2500.0;
  force["z"] = 2500.0;
  force["source_surf"] = false;
  force["angle"] = 0.0;
  force["fx"] = 0.0;
  force["fy"] = 0.0;
  force["fz"] = 0.0;
  force["Ricker"]["factor"] = 1e10;
  force["Ricker"]["tshift"] = 5.0;
  force["Ricker"]["f0"] = 10.0;
  source["force"] = force;
  node["sources"].push_back(source);
  return node;
}();

const static YAML::Node single_moment_tensor_yaml_3d = []() {
  YAML::Node node;
  node["number-of-sources"] = 1;
  YAML::Node source;
  YAML::Node moment_tensor;
  moment_tensor["x"] = 2000.0;
  moment_tensor["y"] = 3000.0;
  moment_tensor["z"] = 2000.0;
  moment_tensor["Mxx"] = 1.0;
  moment_tensor["Myy"] = 1.0;
  moment_tensor["Mzz"] = 0.0;
  moment_tensor["Mxy"] = 1.0;
  moment_tensor["Mxz"] = 0.0;
  moment_tensor["Myz"] = 0.0;
  moment_tensor["Ricker"]["factor"] = 1.0e10;
  moment_tensor["Ricker"]["tshift"] = 30.0;
  moment_tensor["Ricker"]["f0"] = 1.0;
  source["moment-tensor"] = moment_tensor;
  node["sources"].push_back(source);
  return node;
}();

const static YAML::Node multiple_sources_yaml_2d = []() {
  YAML::Node node;
  node["number-of-sources"] = 2;

  // First source - moment tensor
  YAML::Node source1;
  YAML::Node moment_tensor;
  moment_tensor["x"] = 2000.0;
  moment_tensor["z"] = 3000.0;
  moment_tensor["Mxx"] = 1.0;
  moment_tensor["Mzz"] = 1.0;
  moment_tensor["Mxz"] = 0.0;
  moment_tensor["Ricker"]["factor"] = 1.0e10;
  moment_tensor["Ricker"]["tshift"] = 30.0;
  moment_tensor["Ricker"]["f0"] = 1.0;
  source1["moment-tensor"] = moment_tensor;

  // Second source - force
  YAML::Node source2;
  YAML::Node force;
  force["x"] = 2500.0;
  force["z"] = 2500.0;
  force["source_surf"] = false;
  force["angle"] = 0.0;
  force["vx"] = 0.0;
  force["vz"] = 0.0;
  force["Ricker"]["factor"] = 1e10;
  force["Ricker"]["tshift"] = 5.0;
  force["Ricker"]["f0"] = 10.0;
  source2["force"] = force;

  node["sources"].push_back(source1);
  node["sources"].push_back(source2);
  return node;
}();

const static YAML::Node multiple_sources_yaml_3d = []() {
  YAML::Node node;
  node["number-of-sources"] = 2;

  // First source - force
  YAML::Node source1;
  YAML::Node force;
  force["x"] = 2500.0;
  force["y"] = 2500.0;
  force["z"] = 2500.0;
  force["source_surf"] = false;
  force["angle"] = 0.0;
  force["fx"] = 0.0;
  force["fy"] = 0.0;
  force["fz"] = 0.0;
  force["Ricker"]["factor"] = 1e10;
  force["Ricker"]["tshift"] = 5.0;
  force["Ricker"]["f0"] = 10.0;
  source1["force"] = force;

  // Second source - moment tensor
  YAML::Node source2;
  YAML::Node moment_tensor;
  moment_tensor["x"] = 2000.0;
  moment_tensor["y"] = 3000.0;
  moment_tensor["z"] = 2000.0;
  moment_tensor["Mxx"] = 1.0;
  moment_tensor["Myy"] = 1.0;
  moment_tensor["Mzz"] = 0.0;
  moment_tensor["Mxy"] = 1.0;
  moment_tensor["Mxz"] = 0.0;
  moment_tensor["Myz"] = 0.0;
  moment_tensor["Ricker"]["factor"] = 1.0e10;
  moment_tensor["Ricker"]["tshift"] = 30.0;
  moment_tensor["Ricker"]["f0"] = 1.0;
  source2["moment-tensor"] = moment_tensor;

  node["sources"].push_back(source1);
  node["sources"].push_back(source2);
  return node;
}();

class Read2DSourcesYAMLTest
    : public ::testing::TestWithParam<SourceYAMLTestParam2D> {};

TEST_P(Read2DSourcesYAMLTest, ReadYAMLnode) {
  const auto &param = GetParam();

  auto [sources, _t0] =
      specfem::io::read_2d_sources(param.sources_node, nsteps, user_t0, dt,
                                   specfem::simulation::type::forward);

  ASSERT_EQ(sources.size(), param.expected_sources.size());

  for (size_t i = 0; i < sources.size(); ++i) {
    auto source = sources[i];
    auto expected_source = param.expected_sources[i];

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
    IO_TESTS, Read2DSourcesYAMLTest,
    ::testing::Values(SourceYAMLTestParam2D{ "2D YAML Moment Tensor",
                                             single_moment_tensor_yaml_2d,
                                             single_moment_tensor_2d },
                      SourceYAMLTestParam2D{ "2D YAML Force",
                                             single_force_yaml_2d,
                                             single_force_2d },
                      SourceYAMLTestParam2D{ "2D YAML Cosserat Force",
                                             single_cosserat_force_yaml_2d,
                                             single_cosserat_force_2d },
                      SourceYAMLTestParam2D{ "2D YAML Multiple Sources",
                                             multiple_sources_yaml_2d,
                                             multiple_sources_2d }));

class Read3DSourcesYAMLTest
    : public ::testing::TestWithParam<SourceYAMLTestParam3D> {};

TEST_P(Read3DSourcesYAMLTest, ReadYAMLnode) {
  const auto &param = GetParam();

  auto [sources, _t0] =
      specfem::io::read_3d_sources(param.sources_node, nsteps, user_t0, dt,
                                   specfem::simulation::type::forward);

  ASSERT_EQ(sources.size(), param.expected_sources.size());

  for (size_t i = 0; i < sources.size(); ++i) {
    auto source = sources[i];
    auto expected_source = param.expected_sources[i];

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
    IO_TESTS, Read3DSourcesYAMLTest,
    ::testing::Values(SourceYAMLTestParam3D{ "3D YAML Force",
                                             single_force_yaml_3d,
                                             single_force_3d },
                      SourceYAMLTestParam3D{ "3D YAML Moment Tensor",
                                             single_moment_tensor_yaml_3d,
                                             single_moment_tensor_3d },
                      SourceYAMLTestParam3D{ "3D YAML Multiple Sources",
                                             multiple_sources_yaml_3d,
                                             multiple_sources_3d }));
