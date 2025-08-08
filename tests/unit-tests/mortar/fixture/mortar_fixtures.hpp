#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "../../MPI_environment.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/specfem_enums.hpp"
#include "mesh/dim2/adjacency_map/adjacency_map.hpp"
#include "mesh/mesh.hpp"

namespace test_configuration {
/**
 * @brief contains the data of a mesh that any test may wish to use.
 *
 */
struct mesh {
public:
  mesh() {};
  mesh(const YAML::Node &node);
  const specfem::mesh::mesh<specfem::dimension::type::dim2> &get_mesh() const;

  std::string name;
  std::string description;
  std::string database;
  int nspec;
  int nmat;
  double characteristic_length;

  // temporarily store interfaces in file while adjacency_graph is WIP
  std::string interface_file;
  bool interface_fluid_2d;
  bool interface_solid_2d;
  bool interface_fluid_fluid_2d;
  bool interface_fluid_solid_2d;
  bool interface_solid_fluid_2d;
  bool interface_solid_solid_2d;

private:
  specfem::mesh::mesh<specfem::dimension::type::dim2> mesh_instance;
};

} // namespace test_configuration

class MESHES : public ::testing::Test,
               public std::vector<test_configuration::mesh> {
protected:
  MESHES();
};
