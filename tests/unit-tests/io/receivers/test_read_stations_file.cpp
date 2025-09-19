#include "../../Kokkos_Environment.hpp"
#include "enumerations/specfem_enums.hpp"
#include "io/interface.hpp"
#include "specfem/receivers.hpp"
#include "test_receiver_solutions.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <gtest/gtest.h>

/**
 * @brief Parameters for testing receiver reading from files.
 *
 * @tparam DimensionTag
 */
template <specfem::dimension::type DimensionTag> struct ReceiverTestParam {
  std::string testname;
  std::string stationsfilename;
  std::vector<std::shared_ptr<specfem::receivers::receiver<DimensionTag> > >
      expected_receivers;
  type_real angle;
};

/**
 * @brief Stream insertion operator for ReceiverTestParam.
 *
 * @tparam DimensionTag
 * @param os
 * @param params
 * @return std::ostream&
 */
template <specfem::dimension::type DimensionTag>
std::ostream &operator<<(std::ostream &os,
                         const ReceiverTestParam<DimensionTag> &params) {
  os << params.testname;
  return os;
}

using ReceiverTestParam2D = ReceiverTestParam<specfem::dimension::type::dim2>;
using ReceiverTestParam3D = ReceiverTestParam<specfem::dimension::type::dim3>;

class Read2DReceiversTest
    : public ::testing::TestWithParam<ReceiverTestParam2D> {};

TEST_P(Read2DReceiversTest, ReadSTATIONSfile) {
  const auto &param = GetParam();

  auto receivers =
      specfem::io::read_2d_receivers(param.stationsfilename, param.angle);

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
    IO_TESTS, Read2DReceiversTest,
    ::testing::Values(
        ReceiverTestParam2D{ "Empty file",
                             "io/receivers/data/dim2/empty_stations.txt",
                             empty_receivers_2d, 0.0 },
        ReceiverTestParam2D{ "Single receiver",
                             "io/receivers/data/dim2/single_station.txt",
                             single_receiver_2d, 0.0 },
        ReceiverTestParam2D{ "Two receivers",
                             "io/receivers/data/dim2/two_stations.txt",
                             two_receivers_2d, 0.0 },
        ReceiverTestParam2D{ "Three receivers",
                             "io/receivers/data/dim2/three_stations.txt",
                             three_receivers_2d, 0.0 },
        ReceiverTestParam2D{ "Ten receivers",
                             "io/receivers/data/dim2/ten_stations.txt",
                             ten_receivers_2d, 0.0 }));

class Read3DReceiversTest
    : public ::testing::TestWithParam<ReceiverTestParam3D> {};

TEST_P(Read3DReceiversTest, ReadSTATIONSfile) {
  const auto &param = GetParam();

  auto receivers = specfem::io::read_3d_receivers(param.stationsfilename);

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
    IO_TESTS, Read3DReceiversTest,
    ::testing::Values(
        ReceiverTestParam3D{ "Empty file",
                             "io/receivers/data/dim3/empty_stations_3d.txt",
                             empty_receivers_3d, 0.0 },
        ReceiverTestParam3D{ "Single receiver",
                             "io/receivers/data/dim3/single_station_3d.txt",
                             single_receiver_3d, 0.0 },
        ReceiverTestParam3D{ "Two receivers",
                             "io/receivers/data/dim3/two_stations_3d.txt",
                             two_receivers_3d, 0.0 },
        ReceiverTestParam3D{ "Three receivers",
                             "io/receivers/data/dim3/three_stations_3d.txt",
                             three_receivers_3d, 0.0 }));
