#include "sources.hpp"

#include <stdexcept>
#include <string>

namespace {

specfem::enums::source_format parse_format_key(const std::string &key) {
  if (key == "YAML") {
    return specfem::enums::source_format::YAML;
  } else if (key == "CMTSOLUTION") {
    return specfem::enums::source_format::CMTSOLUTION;
  } else if (key == "FORCESOLUTION") {
    return specfem::enums::source_format::FORCESOLUTION;
  } else {
    throw std::runtime_error("Unknown source format key: " + key);
  }
}

} // namespace

specfem::runtime_configuration::sources::sources(const YAML::Node &Node) {

  if (Node.IsScalar()) {
    // Old format: sources: "path/to/file.yaml"
    entries.push_back(
        { specfem::enums::source_format::YAML, Node.as<std::string>() });
  } else if (Node.IsMap()) {
    // New format: sources: { YAML: ..., CMTSOLUTION: ..., ... }
    for (auto it = Node.begin(); it != Node.end(); ++it) {
      const std::string key = it->first.as<std::string>();
      const auto format = parse_format_key(key);
      const YAML::Node &value = it->second;

      if (value.IsScalar()) {
        // Single file: YAML: path/to/file.yaml
        entries.push_back({ format, value.as<std::string>() });
      } else if (value.IsSequence()) {
        // Multiple files: YAML: [file1.yaml, file2.yaml]
        for (std::size_t i = 0; i < value.size(); ++i) {
          entries.push_back({ format, value[i].as<std::string>() });
        }
      } else {
        throw std::runtime_error(
            "Source format value must be a file path (string) or a list of "
            "file paths for key: " +
            key);
      }
    }
  } else {
    throw std::runtime_error(
        "Invalid sources configuration: expected a file path string or a "
        "map of format keys");
  }

  if (entries.empty()) {
    throw std::runtime_error("No source file entries found in configuration");
  }
}
