#pragma once

#include "io/reader.hpp"
#include "io/writer.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Configuration for material property I/O operations.
 *
 * Manages output format, location, and read/write mode settings for
 * property file handling. Creates appropriate reader/writer instances
 * based on configuration parameters.
 */
class property {
public:
  /**
   * @brief Construct property configuration from explicit parameters.
   *
   * @param output_format File format for I/O (e.g., "binary", "ascii")
   * @param output_folder Directory path for property files
   * @param write_mode True for writing, false for reading
   */
  property(const std::string &output_format, const std::string &output_folder,
           const bool write_mode)
      : output_format(output_format), output_folder(output_folder),
        write_mode(write_mode) {}

  /**
   * @brief Construct property configuration from YAML node.
   *
   * @param Node YAML configuration node containing I/O settings
   * @param write_mode True for writing, false for reading
   */
  property(const YAML::Node &Node, const bool write_mode);

  /**
   * @brief Create property writer instance based on configuration.
   *
   * @return Shared pointer to instantiated property writer
   */
  std::shared_ptr<specfem::io::writer> instantiate_property_writer() const;

  /**
   * @brief Create property reader instance based on configuration.
   *
   * @return Shared pointer to instantiated property reader
   */
  std::shared_ptr<specfem::io::reader> instantiate_property_reader() const;

private:
  bool write_mode;           ///< I/O mode: true for writing, false for reading
  std::string output_format; ///< File format (binary/ascii)
  std::string output_folder; ///< Directory path for property files
};
} // namespace runtime_configuration
} // namespace specfem
