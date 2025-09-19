#include "../../Kokkos_Environment.hpp"
#include "enumerations/specfem_enums.hpp"
#include "io/interface.hpp"
#include "specfem/receivers.hpp"
#include "test_receiver_solutions.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

/**
 * @brief Parameters for testing receiver reading from YAML nodes.
 *
 * @tparam DimensionTag
 */
template <specfem::dimension::type DimensionTag> struct ReceiverYAMLTestParam {
  std::string testname;
  YAML::Node stations_node;
  std::vector<std::shared_ptr<specfem::receivers::receiver<DimensionTag> > >
      expected_receivers;
  type_real angle;
};

/**
 * @brief Stream insertion operator for ReceiverYAMLTestParam.
 *
 * @tparam DimensionTag
 * @param os
 * @param params
 * @return std::ostream&
 */
template <specfem::dimension::type DimensionTag>
std::ostream &operator<<(std::ostream &os,
                         const ReceiverYAMLTestParam<DimensionTag> &params) {
  os << params.testname;
  return os;
}

using ReceiverYAMLTestParam2D =
    ReceiverYAMLTestParam<specfem::dimension::type::dim2>;

// YAML node test data for 2D receivers
const static YAML::Node empty_stations_yaml_2d = []() {
  YAML::Node node;
  node["stations"] = YAML::Node(YAML::NodeType::Sequence);
  return node;
}();

const static YAML::Node single_receiver_yaml_2d = []() {
  YAML::Node node;
  YAML::Node station;
  station["network"] = "AA";
  station["station"] = "S0001";
  station["x"] = 300.0;
  station["z"] = 3000.0;
  node["stations"].push_back(station);
  return node;
}();

const static YAML::Node two_receivers_yaml_2d = []() {
  YAML::Node node;
  YAML::Node station1;
  station1["network"] = "AA";
  station1["station"] = "S0001";
  station1["x"] = 300.0;
  station1["z"] = 3000.0;
  node["stations"].push_back(station1);

  YAML::Node station2;
  station2["network"] = "AA";
  station2["station"] = "S0002";
  station2["x"] = 640.0;
  station2["z"] = 3000.0;
  node["stations"].push_back(station2);
  return node;
}();

// YAML node test data for 3D receivers
const static YAML::Node empty_stations_yaml_3d = []() {
  YAML::Node node;
  node["stations"] = YAML::Node(YAML::NodeType::Sequence);
  return node;
}();

const static YAML::Node single_receiver_yaml_3d = []() {
  YAML::Node node;
  YAML::Node station;
  station["network"] = "AA";
  station["station"] = "S0001";
  station["x"] = 300.0;
  station["y"] = 3000.0;
  station["z"] = 2000.0;
  node["stations"].push_back(station);
  return node;
}();

const static YAML::Node two_receivers_yaml_3d = []() {
  YAML::Node node;
  YAML::Node station1;
  station1["network"] = "AA";
  station1["station"] = "S0001";
  station1["x"] = 300.0;
  station1["y"] = 3000.0;
  station1["z"] = 2000.0;
  node["stations"].push_back(station1);

  YAML::Node station2;
  station2["network"] = "AA";
  station2["station"] = "S0002";
  station2["x"] = 640.0;
  station2["y"] = 3000.0;
  station2["z"] = 2000.0;
  node["stations"].push_back(station2);
  return node;
}();

class Read2DReceiversYAMLTest
    : public ::testing::TestWithParam<ReceiverYAMLTestParam2D> {};

TEST_P(Read2DReceiversYAMLTest, ReadYAMLnode) {
  const auto &param = GetParam();

  if (param.expected_receivers.empty()) {
    EXPECT_THROW(
        specfem::io::read_2d_receivers(param.stations_node, param.angle),
        std::runtime_error);
    return;
  }

  auto receivers =
      specfem::io::read_2d_receivers(param.stations_node, param.angle);

  ASSERT_EQ(receivers.size(), param.expected_receivers.size());

  for (size_t i = 0; i < receivers.size(); ++i) {
    auto receiver = receivers[i];
    auto expected_receiver = param.expected_receivers[i];

    EXPECT_EQ(*receiver, *expected_receiver)
        << "Receiver mismatch at index " << i << ":\n"
        << "Expected: " << expected_receiver->print()
        << "\nActual: " << receiver->print();
  }
}

INSTANTIATE_TEST_SUITE_P(
    IO_TESTS, Read2DReceiversYAMLTest,
    ::testing::Values(ReceiverYAMLTestParam2D{ "2D YAML Empty",
                                               empty_stations_yaml_2d,
                                               empty_receivers_2d, 0.0 },
                      ReceiverYAMLTestParam2D{ "2D YAML Single receiver",
                                               single_receiver_yaml_2d,
                                               single_receiver_2d, 0.0 },
                      ReceiverYAMLTestParam2D{ "2D YAML Two receivers",
                                               two_receivers_yaml_2d,
                                               two_receivers_2d, 0.0 }));

using ReceiverYAMLTestParam3D =
    ReceiverYAMLTestParam<specfem::dimension::type::dim3>;

class Read3DReceiversYAMLTest
    : public ::testing::TestWithParam<ReceiverYAMLTestParam3D> {};

TEST_P(Read3DReceiversYAMLTest, ReadYAMLnode) {
  const auto &param = GetParam();

  if (param.expected_receivers.empty()) {
    EXPECT_THROW(specfem::io::read_3d_receivers(param.stations_node),
                 std::runtime_error);
    return;
  }

  auto receivers = specfem::io::read_3d_receivers(param.stations_node);

  ASSERT_EQ(receivers.size(), param.expected_receivers.size());

  for (size_t i = 0; i < receivers.size(); ++i) {
    auto receiver = receivers[i];
    auto expected_receiver = param.expected_receivers[i];

    EXPECT_EQ(*receiver, *expected_receiver)
        << "Receiver mismatch at index " << i << ":\n"
        << "Expected: " << expected_receiver->print()
        << "\nActual: " << receiver->print();
  }
}

INSTANTIATE_TEST_SUITE_P(
    IO_TESTS, Read3DReceiversYAMLTest,
    ::testing::Values(ReceiverYAMLTestParam3D{ "3D YAML Empty",
                                               empty_stations_yaml_3d,
                                               empty_receivers_3d, 0.0 },
                      ReceiverYAMLTestParam3D{ "3D YAML Single receiver",
                                               single_receiver_yaml_3d,
                                               single_receiver_3d, 0.0 },
                      ReceiverYAMLTestParam3D{ "3D YAML Two receivers",
                                               two_receivers_yaml_3d,
                                               two_receivers_3d, 0.0 }));
