#pragma once

#include "mesh/modifiers/modifiers.hpp"

#include "constants.hpp"
#include "enumerations/specfem_enums.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <string>

namespace specfem {
namespace runtime_configuration {
/**
 * @brief class to read mesh modifier information
 *
 */
class mesh_modifiers {
public:
  mesh_modifiers(const YAML::Node &Node) : mesh_modifiers_node(Node) {}

  std::shared_ptr<specfem::mesh::modifiers> instantiate_mesh_modifiers();
  void load_subdivisions(specfem::mesh::modifiers &modifiers);

private:
  YAML::Node mesh_modifiers_node; /// Node that contains receiver information
};

} // namespace runtime_configuration
} // namespace specfem
