#pragma once

#include "../../MPI_environment.hpp"
#include "IO/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "mesh/mesh.hpp"
#include "receiver/receiver.hpp"
#include "source/source.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

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

class MESH : public ::testing::Test {

protected:
  class Iterator {
  public:
    Iterator(test_configuration::Test *p_Test,
             specfem::mesh::mesh<specfem::dimension::type::dim2> *p_mesh)
        : p_Test(p_Test), p_mesh(p_mesh) {}

    std::tuple<test_configuration::Test,
               specfem::mesh::mesh<specfem::dimension::type::dim2> >
    operator*() {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[RUNNING]\033[0m " << p_Test->name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      return std::make_tuple(*p_Test, *p_mesh);
    }

    Iterator &operator++() {
      ++p_Test;
      ++p_mesh;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return p_Test != other.p_Test;
    }

  private:
    test_configuration::Test *p_Test;
    specfem::mesh::mesh<specfem::dimension::type::dim2> *p_mesh;
  };

  MESH();

  Iterator begin() { return Iterator(&Tests[0], &meshes[0]); }

  Iterator end() {
    return Iterator(&Tests[Tests.size()], &meshes[meshes.size()]);
  }

  std::vector<test_configuration::Test> Tests;
  std::vector<specfem::mesh::mesh<specfem::dimension::type::dim2> > meshes;
};
