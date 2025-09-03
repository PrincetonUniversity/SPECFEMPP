#include "specfem/receivers.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(ReceiversTests, DefaultConstructor3D) {
  const std::string network_name = "TEST_NETWORK";
  const std::string station_name = "STA01";
  const type_real x = 100.0;
  const type_real y = 150.0;
  const type_real z = 200.0;

  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      network_name, station_name, x, y, z);

  EXPECT_EQ(receiver.get_network_name(), network_name);
  EXPECT_EQ(receiver.get_station_name(), station_name);
  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_y(), y);
  EXPECT_EQ(receiver.get_z(), z);
}

TEST(ReceiversTests, GetNetworkName3D) {
  const std::string network_name = "GLOBAL";
  const std::string station_name = "STA02";

  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      network_name, station_name, 0.0, 0.0, 0.0);

  EXPECT_EQ(receiver.get_network_name(), network_name);
}

TEST(ReceiversTests, GetStationName3D) {
  const std::string network_name = "TEST";
  const std::string station_name = "STATION_TEST";

  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      network_name, station_name, 0.0, 0.0, 0.0);

  EXPECT_EQ(receiver.get_station_name(), station_name);
}

TEST(ReceiversTests, GetCoordinates3D) {
  const type_real x = 1500.5;
  const type_real y = -500.75;
  const type_real z = -800.25;

  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      "NET", "STA", x, y, z);

  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_y(), y);
  EXPECT_EQ(receiver.get_z(), z);
}

TEST(ReceiversTests, NegativeCoordinates3D) {
  const type_real x = -1000.0;
  const type_real y = -1500.0;
  const type_real z = -2000.0;

  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      "NET", "STA", x, y, z);

  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_y(), y);
  EXPECT_EQ(receiver.get_z(), z);
}

TEST(ReceiversTests, ZeroValues3D) {
  const type_real x = 0.0;
  const type_real y = 0.0;
  const type_real z = 0.0;

  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      "NET", "STA", x, y, z);

  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_y(), y);
  EXPECT_EQ(receiver.get_z(), z);
}

TEST(ReceiversTests, EmptyStrings3D) {
  const std::string network_name = "";
  const std::string station_name = "";

  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      network_name, station_name, 0.0, 0.0, 0.0);

  EXPECT_EQ(receiver.get_network_name(), network_name);
  EXPECT_EQ(receiver.get_station_name(), station_name);
}

TEST(ReceiversTests, DimensionTag3D) {
  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      "NET", "STA", 0.0, 0.0, 0.0);

  EXPECT_EQ(receiver.dimension_tag, specfem::dimension::type::dim3);
}

TEST(ReceiversTests, LargeValues3D) {
  const type_real x = 1e10;
  const type_real y = 2e10;
  const type_real z = 3e10;

  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      "NET", "STA", x, y, z);

  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_y(), y);
  EXPECT_EQ(receiver.get_z(), z);
}

TEST(ReceiversTests, PrintMethod3D) {
  const std::string network_name = "TEST_NET";
  const std::string station_name = "TEST_STA";
  const type_real x = 100.0;
  const type_real y = 200.0;
  const type_real z = 300.0;

  specfem::receivers::receiver<specfem::dimension::type::dim3> receiver(
      network_name, station_name, x, y, z);

  // Test that print method exists and doesn't crash
  std::string output = receiver.print();
  EXPECT_FALSE(output.empty());
}
