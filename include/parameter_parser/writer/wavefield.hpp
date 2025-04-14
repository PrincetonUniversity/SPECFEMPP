#pragma once

#include "enumerations/simulation.hpp"
#include "io/reader.hpp"
#include "io/writer.hpp"
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
   * @param time_interval number of timesteps between wavefield if it is 0, do
   * not write wavefield periodically, if it is -1, the second parameter
   * nsteps_between_samples_by_memory will be used instead.
   * @param time_interval_by_memory automatically determine the number of
   * timesteps between wavefield by the memory value provided, e.g. 100MB, 20GB.
   * @param write_last_step Whether or not to write the final time step.
   */
  wavefield(const std::string output_format, const std::string output_folder,
            const specfem::simulation::type type, const int time_interval,
            const std::string time_interval_by_memory,
            const bool write_last_step)
      : output_format(output_format), output_folder(output_folder),
        simulation_type(type), time_interval(time_interval),
        time_interval_by_memory(time_interval_by_memory),
        write_last_step(write_last_step) {}

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
   * @return std::shared_ptr<specfem::io::writer> Pointer to an instantiated
   * writer object
   */
  std::shared_ptr<specfem::io::writer> instantiate_wavefield_writer() const;

  /**
   * @brief Instantiate a wavefield reader object
   *
   * @return std::shared_ptr<specfem::io::reader> Pointer to an instantiated
   * reader object
   */
  std::shared_ptr<specfem::io::reader> instantiate_wavefield_reader() const;

  inline specfem::simulation::type get_simulation_type() const {
    return this->simulation_type;
  }

private:
  std::string output_format;                 ///< format of output file
  std::string output_folder;                 ///< Path to output folder
  specfem::simulation::type simulation_type; ///< Type of simulation
  int time_interval;                         ///< Number of timesteps between
                                             ///< wavefield
  std::string time_interval_by_memory; ///< Automatically determine the number
                                       ///< of timesteps between wavefield by
                                       ///< the memory value provided, e.g.
                                       ///< 100MB, 20GB.
  bool write_last_step; ///< Whether or not to write the final time step
};
} // namespace runtime_configuration
} // namespace specfem
