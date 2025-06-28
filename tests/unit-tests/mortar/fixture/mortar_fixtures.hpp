#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "enumerations/dimension.hpp"
#include "enumerations/specfem_enums.hpp"
#include "mesh/dim2/adjacency_map/adjacency_map.hpp"

namespace test_configuration {
/**
 * @brief contains the data of a mesh that any test may wish to use.
 *
 */
struct mesh {
public:
  mesh() {};
  mesh(const YAML::Node &node) {
    name = node["name"].as<std::string>();
    description = node["description"].as<std::string>();
    database = node["database"].as<std::string>();
    correct_adjacencies = node["correct-adjacencies"].as<std::string>();
    correct_corner_groups = node["correct-corner-groups"].as<std::string>();
    nspec = node["nspec"].as<int>();
  }

  std::string name;
  std::string description;
  std::string database;
  std::string correct_adjacencies;
  std::string correct_corner_groups;
  int nspec;

  specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
  reference_adjacency_map() const;
  std::vector<std::set<std::pair<int, specfem::enums::boundaries::type> > >
  reference_corner_groups() const;
};
} // namespace test_configuration

class MESHES : public ::testing::Test,
               public std::vector<test_configuration::mesh> {
protected:
  MESHES();
};
