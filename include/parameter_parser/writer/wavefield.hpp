#ifndef _SPECFEM_RUNTIME_CONFIGURATION_WAVEFIELD_HPP
#define _SPECFEM_RUNTIME_CONFIGURATION_WAVEFIELD_HPP

#include "compute/assembly/assembly.hpp"
#include "reader/reader.hpp"
#include "writer/writer.hpp"
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
   * @param assembly SPECFEM++ assembly object
   * @return std::shared_ptr<specfem::writer::writer> Pointer to an instantiated
   * writer object
   */
  std::shared_ptr<specfem::writer::writer> instantiate_wavefield_writer(
      const specfem::compute::assembly &assembly) const;

  /**
   * @brief Instantiate a wavefield reader object
   *
   * @param assembly SPECFEM++ assembly object
   * @return std::shared_ptr<specfem::reader::reader> Pointer to an instantiated
   * reader object
   */
  std::shared_ptr<specfem::reader::reader> instantiate_wavefield_reader(
      const specfem::compute::assembly &assembly) const;

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

#endif /* _SPECFEM_RUNTIME_CONFIGURATION_WAVEFIELD_HPP */
