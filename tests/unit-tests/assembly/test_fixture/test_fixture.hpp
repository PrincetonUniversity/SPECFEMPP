#pragma once

#include "../../MPI_environment.hpp"
#include "IO/interface.hpp"
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
    suffix = Node["suffix"].as<std::string>();
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

  std::string get_suffix() { return suffix; }

  std::string name;
  std::string description;
  std::string suffix;
  test_configuration::database database;
};
} // namespace test_configuration

// ------------------------------------------------------------------------

class ASSEMBLY : public ::testing::Test {

protected:
  class Iterator {
  public:
    Iterator(
        test_configuration::Test *p_Test,
        specfem::mesh::mesh<specfem::dimension::type::dim2> *p_mesh,
        std::vector<std::shared_ptr<specfem::sources::source> > *p_sources,
        std::vector<std::shared_ptr<specfem::receivers::receiver> > *p_stations,
        std::string *p_suffixes, specfem::compute::assembly *p_assembly)
        : p_Test(p_Test), p_mesh(p_mesh), p_sources(p_sources),
          p_stations(p_stations), p_suffixes(p_suffixes),
          p_assembly(p_assembly) {}

    std::tuple<test_configuration::Test,
               specfem::mesh::mesh<specfem::dimension::type::dim2>,
               std::vector<std::shared_ptr<specfem::sources::source> >,
               std::vector<std::shared_ptr<specfem::receivers::receiver> >,
               std::string, specfem::compute::assembly>
    operator*() {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[RUNNING]\033[0m " << p_Test->name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      return std::make_tuple(*p_Test, *p_mesh, *p_sources, *p_stations,
                             *p_suffixes, *p_assembly);
    }

    Iterator &operator++() {
      ++p_Test;
      ++p_mesh;
      ++p_sources;
      ++p_stations;
      ++p_suffixes;
      ++p_assembly;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return p_Test != other.p_Test;
    }

  private:
    test_configuration::Test *p_Test;
    specfem::mesh::mesh<specfem::dimension::type::dim2> *p_mesh;
    std::vector<std::shared_ptr<specfem::sources::source> > *p_sources;
    std::vector<std::shared_ptr<specfem::receivers::receiver> > *p_stations;
    std::string *p_suffixes;
    specfem::compute::assembly *p_assembly;
  };

  ASSEMBLY();

  Iterator begin() {
    return Iterator(&Tests[0], &Meshes[0], &Sources[0], &Stations[0],
                    &suffixes[0], &assemblies[0]);
  }

  Iterator end() {
    return Iterator(&Tests[Tests.size()], &Meshes[Meshes.size()],
                    &Sources[Sources.size()], &Stations[Stations.size()],
                    &suffixes[suffixes.size()], &assemblies[assemblies.size()]);
  }

  std::vector<test_configuration::Test> Tests;
  std::vector<specfem::mesh::mesh<specfem::dimension::type::dim2> > Meshes;
  std::vector<std::vector<std::shared_ptr<specfem::sources::source> > > Sources;
  std::vector<std::vector<std::shared_ptr<specfem::receivers::receiver> > >
      Stations;
  std::vector<std::string> suffixes;
  std::vector<specfem::compute::assembly> assemblies;
};
