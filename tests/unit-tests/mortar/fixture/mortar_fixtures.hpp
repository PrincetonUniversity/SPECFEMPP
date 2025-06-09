#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <yaml-cpp/yaml.h>

// Test configuration
namespace test_configuration {
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
