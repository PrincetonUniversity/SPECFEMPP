#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace test_configuration {
/**
 * @brief contains the data of a mesh that any test may wish to use.
 *
 */
struct mesh {
public:
  mesh() {};
  mesh(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    database = Node["database"].as<std::string>();
  }

  std::string name;
  std::string description;
  std::string database;
};
} // namespace test_configuration

class MESHES : public ::testing::Test,
               public std::vector<test_configuration::mesh> {
protected:
  MESHES();
};
