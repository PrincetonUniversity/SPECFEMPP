#include "specfem/receivers.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(ReceiversTests, DefaultConstructor2D) {
  const std::string network_name = "TEST_NETWORK";
  const std::string station_name = "STA01";
  const type_real x = 100.0;
  const type_real z = 200.0;
  const type_real angle = 45.0;

  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      network_name, station_name, x, z, angle);

  EXPECT_EQ(receiver.get_network_name(), network_name);
  EXPECT_EQ(receiver.get_station_name(), station_name);
  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_z(), z);
  EXPECT_EQ(receiver.get_angle(), angle);
}

TEST(ReceiversTests, GetNetworkName2D) {
  const std::string network_name = "GLOBAL";
  const std::string station_name = "STA02";

  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      network_name, station_name, 0.0, 0.0, 0.0);

  EXPECT_EQ(receiver.get_network_name(), network_name);
}

TEST(ReceiversTests, GetStationName2D) {
  const std::string network_name = "TEST";
  const std::string station_name = "STATION_TEST";

  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      network_name, station_name, 0.0, 0.0, 0.0);

  EXPECT_EQ(receiver.get_station_name(), station_name);
}

TEST(ReceiversTests, GetCoordinates2D) {
  const type_real x = 1500.5;
  const type_real z = -800.25;

  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      "NET", "STA", x, z, 0.0);

  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_z(), z);
}

TEST(ReceiversTests, GetAngle2D) {
  const type_real angle = 90.0;

  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      "NET", "STA", 0.0, 0.0, angle);

  EXPECT_EQ(receiver.get_angle(), angle);
}

TEST(ReceiversTests, NegativeCoordinates2D) {
  const type_real x = -1000.0;
  const type_real z = -2000.0;
  const type_real angle = -45.0;

  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      "NET", "STA", x, z, angle);

  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_z(), z);
  EXPECT_EQ(receiver.get_angle(), angle);
}

TEST(ReceiversTests, ZeroValues2D) {
  const type_real x = 0.0;
  const type_real z = 0.0;
  const type_real angle = 0.0;

  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      "NET", "STA", x, z, angle);

  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_z(), z);
  EXPECT_EQ(receiver.get_angle(), angle);
}

TEST(ReceiversTests, EmptyStrings2D) {
  const std::string network_name = "";
  const std::string station_name = "";

  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      network_name, station_name, 0.0, 0.0, 0.0);

  EXPECT_EQ(receiver.get_network_name(), network_name);
  EXPECT_EQ(receiver.get_station_name(), station_name);
}

TEST(ReceiversTests, DimensionTag2D) {
  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      "NET", "STA", 0.0, 0.0, 0.0);

  EXPECT_EQ(receiver.dimension_tag, specfem::dimension::type::dim2);
}

TEST(ReceiversTests, LargeValues2D) {
  const type_real x = 1e10;
  const type_real z = 1e10;
  const type_real angle = 360.0;

  specfem::receivers::receiver<specfem::dimension::type::dim2> receiver(
      "NET", "STA", x, z, angle);

  EXPECT_EQ(receiver.get_x(), x);
  EXPECT_EQ(receiver.get_z(), z);
  EXPECT_EQ(receiver.get_angle(), angle);
}
