#include "specfem/assembly/receivers/impl/receiver_iterator.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace specfem::assembly::receivers_impl;

// Test SeismogramTypeIterator
TEST(SeismogramTypeIteratorTests, DefaultConstructor) {
  SeismogramTypeIterator iterator;
  EXPECT_EQ(iterator.size(), 0);
  EXPECT_EQ(iterator.begin(), iterator.end());
}

TEST(SeismogramTypeIteratorTests, ConstructorWithTypes) {
  std::vector<specfem::wavefield::type> types = {
    specfem::wavefield::type::displacement, specfem::wavefield::type::velocity,
    specfem::wavefield::type::acceleration
  };

  SeismogramTypeIterator iterator(types);
  EXPECT_EQ(iterator.size(), 3);
  EXPECT_NE(iterator.begin(), iterator.end());
}

TEST(SeismogramTypeIteratorTests, IteratorTraversal) {
  std::vector<specfem::wavefield::type> types = {
    specfem::wavefield::type::displacement, specfem::wavefield::type::velocity
  };

  SeismogramTypeIterator iterator(types);
  auto it = iterator.begin();

  EXPECT_EQ(*it, specfem::wavefield::type::displacement);
  ++it;
  EXPECT_EQ(*it, specfem::wavefield::type::velocity);
  ++it;
  EXPECT_EQ(it, iterator.end());
}

TEST(SeismogramTypeIteratorTests, EmptyIterator) {
  std::vector<specfem::wavefield::type> empty_types;
  SeismogramTypeIterator iterator(empty_types);

  EXPECT_EQ(iterator.size(), 0);
  EXPECT_EQ(iterator.begin(), iterator.end());
}

// Test StationInfo
TEST(StationInfoTests, Constructor) {
  std::vector<specfem::wavefield::type> types = {
    specfem::wavefield::type::displacement
  };

  StationInfo station("NET", "STA01", types);
  EXPECT_EQ(station.network_name, "NET");
  EXPECT_EQ(station.station_name, "STA01");
}

TEST(StationInfoTests, GetSeismogramTypes) {
  std::vector<specfem::wavefield::type> types = {
    specfem::wavefield::type::displacement, specfem::wavefield::type::velocity
  };

  StationInfo station("NET", "STA01", types);
  auto seismo_iterator = station.get_seismogram_types();

  EXPECT_EQ(seismo_iterator.size(), 2);
}

TEST(StationInfoTests, EmptyTypes) {
  std::vector<specfem::wavefield::type> empty_types;
  StationInfo station("NET", "STA01", empty_types);

  auto seismo_iterator = station.get_seismogram_types();
  EXPECT_EQ(seismo_iterator.size(), 0);
}

// Test StationIterator
TEST(StationIteratorTests, DefaultConstructor) {
  StationIterator iterator;
  EXPECT_EQ(iterator.size(), 0);
  EXPECT_EQ(iterator.begin(), iterator.end());
}

TEST(StationIteratorTests, ParameterizedConstructor) {
  std::vector<specfem::wavefield::type> types = {
    specfem::wavefield::type::displacement
  };

  StationIterator iterator(2, types);
  EXPECT_EQ(iterator.size(), 0); // No stations added yet
}

TEST(StationIteratorTests, AddStations) {
  std::vector<specfem::wavefield::type> types = {
    specfem::wavefield::type::displacement
  };

  StationIterator iterator(2, types);

  // Manually add stations by accessing protected members through derived class
  class TestStationIterator : public StationIterator {
  public:
    TestStationIterator(size_t nreceivers,
                        const std::vector<specfem::wavefield::type> &types)
        : StationIterator(nreceivers, types) {}

    void add_station(const std::string &network, const std::string &station) {
      network_names_.push_back(network);
      station_names_.push_back(station);
    }
  };

  TestStationIterator test_iterator(2, types);
  test_iterator.add_station("NET1", "STA01");
  test_iterator.add_station("NET2", "STA02");

  EXPECT_EQ(test_iterator.size(), 2);
  EXPECT_NE(test_iterator.begin(), test_iterator.end());
}

TEST(StationIteratorTests, IteratorDereference) {
  std::vector<specfem::wavefield::type> types = {
    specfem::wavefield::type::displacement, specfem::wavefield::type::velocity
  };

  class TestStationIterator : public StationIterator {
  public:
    TestStationIterator(size_t nreceivers,
                        const std::vector<specfem::wavefield::type> &types)
        : StationIterator(nreceivers, types) {}

    void add_station(const std::string &network, const std::string &station) {
      network_names_.push_back(network);
      station_names_.push_back(station);
    }
  };

  TestStationIterator test_iterator(1, types);
  test_iterator.add_station("TEST_NET", "TEST_STA");

  auto it = test_iterator.begin();
  StationInfo station = *it;

  EXPECT_EQ(station.network_name, "TEST_NET");
  EXPECT_EQ(station.station_name, "TEST_STA");

  auto seismo_types = station.get_seismogram_types();
  EXPECT_EQ(seismo_types.size(), 2);
}

TEST(StationIteratorTests, IteratorIncrement) {
  std::vector<specfem::wavefield::type> types = {
    specfem::wavefield::type::displacement
  };

  class TestStationIterator : public StationIterator {
  public:
    TestStationIterator(size_t nreceivers,
                        const std::vector<specfem::wavefield::type> &types)
        : StationIterator(nreceivers, types) {}

    void add_station(const std::string &network, const std::string &station) {
      network_names_.push_back(network);
      station_names_.push_back(station);
    }
  };

  TestStationIterator test_iterator(2, types);
  test_iterator.add_station("NET1", "STA01");
  test_iterator.add_station("NET2", "STA02");

  auto it = test_iterator.begin();
  StationInfo station1 = *it;
  EXPECT_EQ(station1.station_name, "STA01");

  ++it;
  StationInfo station2 = *it;
  EXPECT_EQ(station2.station_name, "STA02");

  ++it;
  EXPECT_EQ(it, test_iterator.end());
}
