#pragma once

#include "specfem/enums.hpp"
#include "yaml-cpp/yaml.h"
#include <string>
#include <vector>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Class to read and manage source configuration.
 *
 * Parses the "sources" node from specfem_config.yaml and produces a list of
 * source file entries with their formats. Supports both the old scalar format
 * (bare path to a YAML file) and the new map format with explicit format keys.
 *
 * Old format (backward compatible):
 * @code{.yaml}
 * sources: "path/to/sources.yaml"
 * @endcode
 *
 * New format:
 * @code{.yaml}
 * sources:
 *   YAML: sources.yaml
 *   CMTSOLUTION:
 *     - CMTSOLUTION_event1
 *     - CMTSOLUTION_event2
 * @endcode
 */
class sources {
public:
  sources(const YAML::Node &Node);

  /**
   * @brief Get the parsed source file entries.
   *
   * @return const reference to vector of source_file_entry
   */
  const std::vector<specfem::enums::source_file_entry> &
  get_source_entries() const {
    return entries;
  }

protected:
  std::vector<specfem::enums::source_file_entry> entries;
};

namespace sources_impl {

/**
 * @brief Parse a source format string key into the corresponding enum.
 *
 * @param key Format key string (e.g., "YAML", "CMTSOLUTION", "FORCESOLUTION")
 * @return Corresponding source_format enum value
 */
specfem::enums::source_format parse_format_key(const std::string &key);

} // namespace sources_impl

} // namespace runtime_configuration
} // namespace specfem
