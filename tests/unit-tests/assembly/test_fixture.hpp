#pragma once

#include "../MPI_environment.hpp"
#include "IO/mesh/read_mesh.hpp"
#include "IO/receivers/read_receivers.hpp"
#include "IO/sources/read_sources.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/specfem_enums.hpp"
#include "mesh/mesh.hpp"
#include "quadrature/quadratures.hpp"
#include "receiver/receiver.hpp"
#include "source/source.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

template <bool using_simd>
KOKKOS_FUNCTION
    specfem::point::index<specfem::dimension::type::dim2, using_simd>
    get_index(const int ielement, const int num_elements, const int iz,
              const int ix);

template <>
KOKKOS_FUNCTION specfem::point::index<specfem::dimension::type::dim2, true>
get_index<true>(const int ielement, const int num_elements, const int iz,
                const int ix) {
  return specfem::point::simd_index<specfem::dimension::type::dim2>(
      ielement, num_elements, iz, ix);
}

template <>
KOKKOS_FUNCTION specfem::point::index<specfem::dimension::type::dim2, false>
get_index<false>(const int ielement, const int num_elements, const int iz,
                 const int ix) {
  return specfem::point::index<specfem::dimension::type::dim2>(ielement, iz,
                                                               ix);
}

// ------------------------------------------------------------------------
// Test configuration
namespace test_configuration {
struct database {
public:
  database() : mesh(""), sources(""), stations(""){};
  database(const YAML::Node &Node) {
    mesh = Node["mesh"].as<std::string>();
    sources = Node["sources"].as<std::string>();
    stations = Node["stations"].as<std::string>();
  }

  std::tuple<std::string, std::string, std::string> get_databases() {
    return std::make_tuple(mesh, sources, stations);
  }

  std::string mesh;
  std::string sources;
  std::string stations;
};

struct Test {
public:
  Test(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    YAML::Node databases = Node["databases"];
    try {
      database = test_configuration::database(databases);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Error in test configuration: " + name + "\n" +
                               e.what());
    }
    return;
  }

  std::tuple<std::string, std::string, std::string> get_databases() {
    return database.get_databases();
  }

  std::string name;
  std::string description;
  test_configuration::database database;
};
} // namespace test_configuration

// ------------------------------------------------------------------------
// Reading test config

void parse_test_config(const YAML::Node &yaml,
                       std::vector<test_configuration::Test> &tests) {
  YAML::Node all_tests = yaml["Tests"];
  assert(all_tests.IsSequence());

  for (auto N : all_tests)
    tests.push_back(test_configuration::Test(N));

  return;
}

// ------------------------------------------------------------------------

class ASSEMBLY : public ::testing::Test {

protected:
  class Iterator {
  public:
    Iterator(test_configuration::Test *p_Test,
             specfem::compute::assembly *p_assembly)
        : p_Test(p_Test), p_assembly(p_assembly) {}

    std::tuple<test_configuration::Test, specfem::compute::assembly>
    operator*() {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[RUNNING]\033[0m " << p_Test->name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      return std::make_tuple(*p_Test, *p_assembly);
    }

    Iterator &operator++() {
      ++p_Test;
      ++p_assembly;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return p_Test != other.p_Test;
    }

  private:
    test_configuration::Test *p_Test;
    specfem::compute::assembly *p_assembly;
  };

  ASSEMBLY() {

    std::string config_filename =
        "../../../tests/unit-tests/assembly/test_config.yaml";
    parse_test_config(YAML::LoadFile(config_filename), Tests);

    specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

    const auto quadrature = []() {
      specfem::quadrature::gll::gll gll{};
      return specfem::quadrature::quadratures(gll);
    }();

    for (auto &Test : Tests) {
      const auto [database_file, sources_file, stations_file] =
          Test.get_databases();
      specfem::mesh::mesh mesh = specfem::IO::read_mesh(database_file, mpi);

      const auto [sources, t0] = specfem::IO::read_sources(
          sources_file, 0, 0, 0, specfem::simulation::type::forward);

      const auto receivers = specfem::IO::read_receivers(stations_file, 0);

      std::vector<specfem::enums::seismogram::type> seismogram_types = {
        specfem::enums::seismogram::type::displacement
      };

      assemblies.push_back(specfem::compute::assembly(
          mesh, quadrature, sources, receivers, seismogram_types, t0, 0, 0, 0,
          specfem::simulation::type::forward));
    }
  }

  Iterator begin() { return Iterator(&Tests[0], &assemblies[0]); }

  Iterator end() {
    return Iterator(&Tests[Tests.size()], &assemblies[assemblies.size()]);
  }

  std::vector<test_configuration::Test> Tests;
  std::vector<specfem::compute::assembly> assemblies;
};
