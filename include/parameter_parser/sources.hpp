#pragma once

#include "enumerations/specfem_enums.hpp"
#include "yaml-cpp/yaml.h"
#include <string>

namespace specfem {
namespace runtime_configuration {
/**
 * @brief class to read source information
 *
 */
class sources {
public:
  sources(const YAML::Node &Node) { source_node = Node; };

  /**
   * @brief Get the sources
   *
   * @return YAML::Node describing the sources
   */
  YAML::Node get_sources() const { return source_node; }

protected:
  YAML::Node source_node; /// Node that contains sources information
};
} // namespace runtime_configuration
} // namespace specfem
