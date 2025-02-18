#pragma once

#include "enumerations/dimension.hpp"
#include "mesh/modifiers/modifiers.hpp"

#include "yaml-cpp/yaml.h"
#include <memory>

namespace specfem {
namespace runtime_configuration {
/**
 * @brief class to read mesh modifier information
 *
 */
class mesh_modifiers {
public:
  mesh_modifiers(const YAML::Node &Node) : mesh_modifiers_node(Node) {}

  template <specfem::dimension::type DimensionType>
  std::shared_ptr<specfem::mesh::modifiers<DimensionType> >
  instantiate_mesh_modifiers();
  template <specfem::dimension::type DimensionType>
  void load_subdivisions(specfem::mesh::modifiers<DimensionType> &modifiers);

private:
  YAML::Node mesh_modifiers_node; /// Node that contains receiver information
};

} // namespace runtime_configuration
} // namespace specfem

#include "parameter_parser/mesh_modifiers.tpp"
