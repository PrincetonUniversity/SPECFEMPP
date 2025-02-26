#pragma once

#include "IO/reader.hpp"
#include "IO/writer.hpp"
#include "enumerations/simulation.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Wavefield configuration class is used to instantiate wavefield writers
 *
 */
class wavefield {

public:
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new wavefield configuration object
   *
   * @param output_format Output wavefield file format
   * @param output_folder Path to folder location where wavefield will be stored
   * @param type Type of simulation (forward or adjoint)
   */
  wavefield(const std::string output_format, const std::string output_folder,
            const specfem::simulation::type type)
      : output_format(output_format), output_folder(output_folder),
        simulation_type(type) {}

  /**
   * @brief Construct a new wavefield configuration object from YAML node
   *
   * @param Node YAML node describing the wavefield writer
   * @param type Type of simulation (forward or adjoint)
   */
  wavefield(const YAML::Node &Node, const specfem::simulation::type type);
  ///@}

  /**
   * @brief Instantiate a wavefield writer object
   *
   * @return std::shared_ptr<specfem::IO::writer> Pointer to an instantiated
   * writer object
   */
  std::shared_ptr<specfem::IO::writer> instantiate_wavefield_writer() const;

  /**
   * @brief Instantiate a wavefield reader object
   *
   * @return std::shared_ptr<specfem::IO::reader> Pointer to an instantiated
   * reader object
   */
  std::shared_ptr<specfem::IO::reader> instantiate_wavefield_reader() const;

  inline specfem::simulation::type get_simulation_type() const {
    return this->simulation_type;
  }

private:
  std::string output_format;                 ///< format of output file
  std::string output_folder;                 ///< Path to output folder
  specfem::simulation::type simulation_type; ///< Type of simulation
};
} // namespace runtime_configuration
} // namespace specfem
