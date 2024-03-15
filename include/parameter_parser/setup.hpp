#ifndef _PARAMETER_SETUP_HPP
#define _PARAMETER_SETUP_HPP

#include "database_configuration.hpp"
#include "header.hpp"
#include "parameter_parser/solver/interface.hpp"
#include "quadrature.hpp"
#include "reader/reader.hpp"
#include "receivers.hpp"
#include "run_setup.hpp"
#include "specfem_setup.hpp"
#include "time_scheme/interface.hpp"
#include "writer/seismogram.hpp"
#include "writer/wavefield.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <tuple>

namespace specfem {
namespace runtime_configuration {
/**
 * Setup class is used to read the YAML file parameter file.
 *
 * Setup class is also used to instantiate the simulation i.e. instantiate
 * quadrature objects, instantiate solver objects.
 *
 */
class setup {

public:
  /**
   * @brief Construct a new setup object
   *
   * @param parameter_file Path to a configuration YAML file
   * @param default_file Path to a YAML file to be used to instantiate default
   * parameters
   */
  setup(const std::string &parameter_file, const std::string &default_file);
  /**
   * @brief Instantiate quadrature objects in x and z dimensions
   *
   * @return std::tuple<specfem::quadrature::quadrature,
   * specfem::quadrature::quadrature> Quadrature objects in x and z dimensions
   */
  specfem::quadrature::quadratures instantiate_quadrature() const {
    return this->quadrature->instantiate();
  }
  // /**
  //  * @brief Instantiate the Timescheme
  //  *
  //  * @return specfem::TimeScheme::TimeScheme* Pointer to the TimeScheme
  //  object
  //  * used in the solver algorithm
  //  */
  std::shared_ptr<specfem::time_scheme::time_scheme>
  instantiate_timescheme() const {
    return this->time_scheme->instantiate(
        this->receivers->get_nstep_between_samples());
  }
  // /**
  //  * @brief Update simulation start time.
  //  *
  //  * If user has not defined start time then we need to update the simulation
  //  * start time based on source frequencies and time shift
  //  *
  //  * @note This might be specific to only time-marching solvers
  //  *
  //  * @param t0 Simulation start time
  //  */
  void update_t0(type_real t0) { this->time_scheme->update_t0(t0); }
  /**
   * @brief Log the header and description of the simulation
   */
  std::string print_header(
      const std::chrono::time_point<std::chrono::high_resolution_clock> now);

  /**
   * @brief Get delta time value
   *
   * @return type_real
   */
  type_real get_dt() const { return time_scheme->get_dt(); }

  /**
   * @brief Get the path to mesh database and source yaml file
   *
   * @return std::tuple<std::string, std::string> std::tuple specifying the path
   * to mesh database and source yaml file
   */
  std::tuple<std::string, std::string> get_databases() const {
    return databases->get_databases();
  }

  /**
   * @brief Get the path to stations file
   *
   * @return std::string path to stations file
   */
  std::string get_stations_file() const {
    return this->receivers->get_stations_file();
  }

  /**
   * @brief Get the angle of receivers
   *
   * @return type_real angle of the receiver
   */
  type_real get_receiver_angle() const { return this->receivers->get_angle(); }

  /**
   * @brief Get the types of siesmograms to be calculated
   *
   * @return std::vector<specfem::seismogram::type> Types of seismograms to be
   * calculated
   */
  std::vector<specfem::enums::seismogram::type> get_seismogram_types() const {
    return this->receivers->get_seismogram_types();
  }

  /**
   * @brief Instantiate a seismogram writer object
   *
   * @param receivers Pointer to specfem::compute::receivers struct
   used
   * to instantiate the writer
   * @return specfem::writer::writer* Pointer to an instantiated writer
   object
   */
  std::shared_ptr<specfem::writer::writer> instantiate_seismogram_writer(
      const specfem::compute::assembly &assembly) const {
    if (this->seismogram) {
      return this->seismogram->instantiate_seismogram_writer(
          assembly.receivers, this->time_scheme->get_dt(),
          this->time_scheme->get_t0(),
          this->receivers->get_nstep_between_samples());
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<specfem::writer::writer> instantiate_wavefield_writer(
      const specfem::compute::assembly &assembly) const {
    if (this->wavefield) {
      return this->wavefield->instantiate_wavefield_writer(assembly);
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<specfem::reader::reader> instantiate_wavefield_reader(
      const specfem::compute::assembly &assembly) const {
    if (this->wavefield) {
      return this->wavefield->instantiate_wavefield_reader(assembly);
    } else {
      return nullptr;
    }
  }

  inline specfem::simulation::type get_simulation_type() const {
    // Wavefield needs to always be defined for any simulation other than
    // forward
    if (!this->wavefield) {
      return specfem::simulation::type::forward;
    }
    return this->wavefield->get_simulation_type();
  }

  template <typename qp_type>
  std::shared_ptr<specfem::solver::solver> instantiate_solver(
      const specfem::compute::assembly &assembly,
      std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const qp_type &quadrature) const {
    return this->solver->instantiate(assembly, time_scheme, quadrature);
  }

  int get_nsteps() const { return this->time_scheme->get_nsteps(); }

private:
  std::unique_ptr<specfem::runtime_configuration::header> header; ///< Pointer
                                                                  ///< to header
                                                                  ///< object
  std::unique_ptr<specfem::runtime_configuration::time_scheme::time_scheme>
      time_scheme; ///< Pointer to solver
                   ///< object
  std::unique_ptr<specfem::runtime_configuration::run_setup>
      run_setup; ///< Pointer to
                 ///< run_setup object
  std::unique_ptr<specfem::runtime_configuration::quadrature>
      quadrature; ///< Pointer to
                  ///< quadrature object
  std::unique_ptr<specfem::runtime_configuration::receivers>
      receivers; ///< Pointer to receivers object
  std::unique_ptr<specfem::runtime_configuration::seismogram>
      seismogram; ///< Pointer to
                  ///< seismogram object
  std::unique_ptr<specfem::runtime_configuration::wavefield>
      wavefield; ///< Pointer to
                 ///< wavefield object
  std::unique_ptr<specfem::runtime_configuration::database_configuration>
      databases; ///< Get database filenames
  std::unique_ptr<specfem::runtime_configuration::solver::solver>
      solver; ///< Pointer to solver object
};
} // namespace runtime_configuration
} // namespace specfem

#endif
