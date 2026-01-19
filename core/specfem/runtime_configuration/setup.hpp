#pragma once

#include "database_configuration.hpp"
#include "elastic_wave.hpp"
#include "electromagnetic_wave.hpp"
#include "header.hpp"
#include "io/reader.hpp"
#include "quadrature.hpp"
#include "receivers.hpp"
#include "run_setup.hpp"
#include "solver.hpp"
#include "solver.tpp"
#include "sources.hpp"
#include "specfem_setup.hpp"
#include "time_scheme.hpp"
#include "writer/kernel.hpp"
#include "writer/plot_wavefield.hpp"
#include "writer/property.hpp"
#include "writer/seismogram.hpp"
#include "writer/wavefield.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <tuple>

namespace specfem {

/**
 * @brief Runtime configuration management for SPECFEM simulations.
 *
 * Contains classes for parsing YAML parameter files and managing simulation
 * configuration including solvers, time schemes, I/O handlers, sources,
 * receivers, and material properties. Provides type-safe configuration
 * objects that bridge parameter files with simulation components.
 *
 * @see specfem::runtime_configuration::setup Implements the holds all the data
 * provided by the YAML file as C++ objects for each YAML section.
 */
namespace runtime_configuration {

/**
 * @brief Main configuration manager for SPECFEM simulations.
 *
 * Parses YAML parameter files and instantiates simulation components
 * including solvers, quadrature, time schemes, and I/O handlers.
 * Central orchestrator for simulation setup and configuration.
 */
class setup {

public:
  /**
   * @brief Construct a new setup object
   *
   * @param parameter_file Path to a configuration YAML file
   * @param default_file Path to a YAML file to be used to instantiate default
   * parameters
   * @param binding_python Flag to indicate if the setup is being used in a
   * pybind environment
   */
  setup(const std::string &parameter_file, const std::string &default_file);
  /**
   * @brief Construct a new setup object
   *
   * @param parameter_dict Configuration YAML Node
   * @param default_dict YAML Node to be used to instantiate default parameters
   */
  setup(const YAML::Node &parameter_dict, const YAML::Node &default_dict);
  /**
   * @brief Instantiate quadrature objects in x and z dimensions
   *
   * @return std::tuple<specfem::quadrature::quadrature,
   * specfem::quadrature::quadrature> Quadrature objects in x and z dimensions
   */
  specfem::quadrature::quadratures instantiate_quadrature() const {
    return this->quadrature->instantiate();
  }
  /**
   * @brief Instantiate the Timescheme
   *
   * @tparam AssemblyFields Assembly fields type (dimension-agnostic)
   * @param fields Assembly fields to link with the timescheme
   * @return specfem::time_scheme::time_scheme* Pointer to the TimeScheme
   object
   * used in the solver algorithm
   */
  template <typename AssemblyFields>
  std::shared_ptr<specfem::time_scheme::time_scheme>
  instantiate_timescheme(AssemblyFields &fields) const {
    return this->time_scheme->instantiate(
        fields, this->receivers->get_nstep_between_samples());
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
  /**
   * @brief Update simulation start time.
   *
   * @param t0 New simulation start time
   */
  void update_t0(type_real t0) { this->time_scheme->update_t0(t0); }

  /**
   * @brief Get simulation start time.
   *
   * @return Current simulation start time
   */
  type_real get_t0() const { return this->time_scheme->get_t0(); }

  /**
   * @brief Get the type of the elastic wave
   *
   * @return specfem::enums::elastic_wave Type of the elastic wave
   */
  inline specfem::enums::elastic_wave get_elastic_wave_type() const {
    return this->elastic_wave->get_elastic_wave_type();
  }

  /**
   * @brief Get the type of the electromagnetic wave
   *
   * @return specfem::enums::electromagnetic_wave Type of the electromagnetic
   * wave
   */
  inline specfem::enums::electromagnetic_wave
  get_electromagnetic_wave_type() const {
    return this->electromagnetic_wave->get_electromagnetic_wave_type();
  }

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
  std::string get_databases() const { return databases->get_databases(); }

  /**
   * @brief Get the sources YAML object
   *
   * @return YAML::Node YAML node describing the sources
   */
  YAML::Node get_sources() const { return this->sources->get_sources(); }

  /**
   * @brief Get the path to stations file
   *
   * @return std::string path to stations file
   */
  YAML::Node get_stations() const { return this->receivers->get_stations(); }

  /**
   * @brief Get the angle of receivers
   *
   * @return type_real angle of the receiver
   */
  type_real get_receiver_angle() const { return this->receivers->get_angle(); }

  /**
   * @brief Get the types of seismograms to be calculated
   *
   * @return std::vector<specfem::seismogram::type> Types of seismograms to be
   * calculated
   */
  std::vector<specfem::wavefield::type> get_seismogram_types() const {
    return this->receivers->get_seismogram_types();
  }

  /**
   * @brief Instantiate a seismogram writer object
   *
   * to instantiate the writer
   * @return specfem::io::writer* Pointer to an instantiated writer
   object
   */
  std::shared_ptr<specfem::io::writer> instantiate_seismogram_writer() const {
    if (this->seismogram) {
      return this->seismogram->instantiate_seismogram_writer(
          this->get_elastic_wave_type(), this->get_electromagnetic_wave_type(),
          this->time_scheme->get_dt(), this->time_scheme->get_t0(),
          this->receivers->get_nstep_between_samples());
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Get number of samples between seismogram recordings
   *
   * @return int number of samples between seismogram recordings
   */
  int get_nstep_between_samples() const {
    return this->receivers->get_nstep_between_samples();
  }

  /**
   * @brief Get the maximum seismogram step
   *
   * @return int Maximum seismogram step
   */
  int get_max_seismogram_step() const {
    return get_nsteps() / get_nstep_between_samples();
  }

  /**
   * @brief Create wavefield writer for periodic output.
   *
   * @tparam DimensionTag Spatial dimension (2D/3D)
   * @return Shared pointer to wavefield writer task or nullptr if not
   * configured
   */
  template <specfem::dimension::type DimensionTag>
  std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag> >
  instantiate_wavefield_writer() const {
    if (this->wavefield) {
      return this->wavefield
          ->template instantiate_wavefield_writer<DimensionTag>();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Create wavefield reader for loading saved wavefields.
   *
   * @tparam DimensionTag Spatial dimension (2D/3D)
   * @return Shared pointer to wavefield reader task or nullptr if not
   * configured
   */
  template <specfem::dimension::type DimensionTag>
  std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag> >
  instantiate_wavefield_reader() const {
    if (this->wavefield) {
      return this->wavefield
          ->template instantiate_wavefield_reader<DimensionTag>();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Create 2D wavefield plotter for visualization.
   *
   * @param assembly 2D assembly containing mesh and field information
   * @param dt Time step size

   * @return Shared pointer to 2D wavefield plotter or nullptr if not configured
   */
  std::shared_ptr<
      specfem::periodic_tasks::periodic_task<specfem::dimension::type::dim2> >
  instantiate_wavefield_plotter(
      const specfem::assembly::assembly<specfem::dimension::type::dim2>
          &assembly,
      const type_real &dt) const {
    if (this->plot_wavefield) {
      return this->plot_wavefield->instantiate_wavefield_plotter(assembly, dt);
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Create 3D wavefield plotter for visualization.
   *
   * @param assembly 3D assembly containing mesh and field information
   * @param dt Time step size
   * @return Shared pointer to 3D wavefield plotter or nullptr if not configured
   */
  std::shared_ptr<
      specfem::periodic_tasks::periodic_task<specfem::dimension::type::dim3> >
  instantiate_wavefield_plotter(
      const specfem::assembly::assembly<specfem::dimension::type::dim3>
          &assembly,
      const type_real &dt) const {
    if (this->plot_wavefield) {
      return this->plot_wavefield->instantiate_wavefield_plotter(assembly, dt);
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Create property reader for loading material properties.
   *
   * @return Shared pointer to property reader or nullptr if not configured
   */
  std::shared_ptr<specfem::io::reader> instantiate_property_reader() const {
    if (this->property) {
      return this->property->instantiate_property_reader();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Create property writer for saving material properties.
   *
   * @return Shared pointer to property writer or nullptr if not configured
   */
  std::shared_ptr<specfem::io::writer> instantiate_property_writer() const {
    if (this->property) {
      return this->property->instantiate_property_writer();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Create kernel writer for sensitivity kernel output.
   *
   * @return Shared pointer to kernel writer or nullptr if not configured
   */
  std::shared_ptr<specfem::io::writer> instantiate_kernel_writer() const {
    if (this->kernel) {
      return this->kernel->instantiate_kernel_writer();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Get simulation type configuration.
   *
   * @return Current simulation type (forward/adjoint/combined)
   */
  inline specfem::simulation::type get_simulation_type() const {
    return this->solver->get_simulation_type();
  }

  /**
   * @brief Create solver instance with specified parameters.
   *
   * @tparam NGLL Number of Gauss-Lobatto-Legendre points per element dimension
   * @tparam DimensionTag Spatial dimension (2D/3D)
   * @param dt Time step size
   * @param assembly Assembly containing mesh and field data
   * @param time_scheme Time integration scheme
   * @param tasks Periodic tasks to execute during simulation
   * @return Shared pointer to configured solver
   */
  template <int NGLL, specfem::dimension::type DimensionTag>
  std::shared_ptr<specfem::solver::solver> instantiate_solver(
      const type_real dt,
      const specfem::assembly::assembly<DimensionTag> &assembly,
      std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<DimensionTag> > > &tasks)
      const {
    return this->solver->instantiate<NGLL, DimensionTag>(dt, assembly,
                                                         time_scheme, tasks);
  }

  /**
   * @brief Get total number of time steps.
   *
   * @return Total simulation time steps
   */
  int get_nsteps() const { return this->time_scheme->get_nsteps(); }

  /**
   * @brief Check if boundary values need allocation.
   *
   * Required for adjoint simulations and combined simulation types.
   *
   * @return True if boundary values should be allocated
   */
  bool allocate_boundary_values() const {
    return (
        ((this->wavefield != nullptr) &&
         (this->wavefield->is_for_adjoint_simulations())) ||
        (this->get_simulation_type() == specfem::simulation::type::combined));
  }

  /**
   * @brief Get the header object
   * @return header Header object
   *
   */
  specfem::runtime_configuration::header get_header() const {
    return *(this->header);
  }

private:
  std::unique_ptr<specfem::runtime_configuration::header>
      header; ///< Simulation header configuration
  std::unique_ptr<specfem::runtime_configuration::elastic_wave>
      elastic_wave; ///< Elastic wave type configuration
  std::unique_ptr<specfem::runtime_configuration::electromagnetic_wave>
      electromagnetic_wave; ///< Electromagnetic wave configuration
  std::unique_ptr<specfem::runtime_configuration::time_scheme>
      time_scheme; ///< Time stepping scheme configuration
  std::unique_ptr<specfem::runtime_configuration::run_setup>
      run_setup; ///< Simulation run configuration
  std::unique_ptr<specfem::runtime_configuration::quadrature>
      quadrature; ///< Numerical quadrature configuration
  std::unique_ptr<specfem::runtime_configuration::receivers>
      receivers; ///< Seismic receiver configuration
  std::unique_ptr<specfem::runtime_configuration::sources>
      sources; ///< Seismic source configuration
  std::unique_ptr<specfem::runtime_configuration::seismogram>
      seismogram; ///< Seismogram output configuration
  std::unique_ptr<specfem::runtime_configuration::wavefield>
      wavefield; ///< Wavefield I/O configuration
  std::unique_ptr<specfem::runtime_configuration::plot_wavefield>
      plot_wavefield; ///< Wavefield plotting configuration
  std::unique_ptr<specfem::runtime_configuration::kernel>
      kernel; ///< Kernel output configuration
  std::unique_ptr<specfem::runtime_configuration::property>
      property; ///< Property I/O configuration
  std::unique_ptr<specfem::runtime_configuration::database_configuration>
      databases; ///< Database file path configuration
  std::unique_ptr<specfem::runtime_configuration::solver>
      solver; ///< Solver algorithm configuration
};
} // namespace runtime_configuration
} // namespace specfem
