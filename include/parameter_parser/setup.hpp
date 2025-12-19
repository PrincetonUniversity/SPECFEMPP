#ifndef _PARAMETER_SETUP_HPP
#define _PARAMETER_SETUP_HPP

#include "database_configuration.hpp"
#include "elastic_wave.hpp"
#include "electromagnetic_wave.hpp"
#include "header.hpp"
#include "io/reader.hpp"
#include "parameter_parser/solver/interface.hpp"
#include "quadrature.hpp"
#include "receivers.hpp"
#include "run_setup.hpp"
#include "sources.hpp"
#include "specfem_setup.hpp"
#include "time_scheme/interface.hpp"
#include "writer/kernel.hpp"
#include "writer/plot_wavefield.hpp"
#include "writer/property.hpp"
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
  void update_t0(type_real t0) { this->time_scheme->update_t0(t0); }

  type_real get_t0() const { return this->time_scheme->get_t0(); }
  /**
   * @brief Log the header and description of the simulation
   */
  std::string
  print_header(const std::chrono::time_point<std::chrono::system_clock> now);

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

  std::shared_ptr<
      specfem::periodic_tasks::periodic_task<specfem::dimension::type::dim2> >
  instantiate_wavefield_plotter(
      const specfem::assembly::assembly<specfem::dimension::type::dim2>
          &assembly,
      const type_real &dt, specfem::MPI::MPI *mpi) const {
    if (this->plot_wavefield) {
      return this->plot_wavefield->instantiate_wavefield_plotter(assembly, dt,
                                                                 mpi);
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<
      specfem::periodic_tasks::periodic_task<specfem::dimension::type::dim3> >
  instantiate_wavefield_plotter(
      const specfem::assembly::assembly<specfem::dimension::type::dim3>
          &assembly,
      const type_real &dt, specfem::MPI::MPI *mpi) const {
    if (this->plot_wavefield) {
      return this->plot_wavefield->instantiate_wavefield_plotter(assembly, dt,
                                                                 mpi);
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<specfem::io::reader> instantiate_property_reader() const {
    if (this->property) {
      return this->property->instantiate_property_reader();
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<specfem::io::writer> instantiate_property_writer() const {
    if (this->property) {
      return this->property->instantiate_property_writer();
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<specfem::io::writer> instantiate_kernel_writer() const {
    if (this->kernel) {
      return this->kernel->instantiate_kernel_writer();
    } else {
      return nullptr;
    }
  }

  inline specfem::simulation::type get_simulation_type() const {
    return this->solver->get_simulation_type();
  }

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

  int get_nsteps() const { return this->time_scheme->get_nsteps(); }

  bool allocate_boundary_values() const {
    return (
        ((this->wavefield != nullptr) &&
         (this->wavefield->is_for_adjoint_simulations())) ||
        (this->get_simulation_type() == specfem::simulation::type::combined));
  }

private:
  std::unique_ptr<specfem::runtime_configuration::header> header; ///< Pointer
                                                                  ///< to header
                                                                  ///< object
  std::unique_ptr<specfem::runtime_configuration::elastic_wave>
      elastic_wave; ///< Pointer to elastic wave object
  std::unique_ptr<specfem::runtime_configuration::electromagnetic_wave>
      electromagnetic_wave; ///< Pointer to electromagnetic wave object
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
  std::unique_ptr<specfem::runtime_configuration::sources>
      sources; ///< Pointer
               ///< to
               ///< receivers
               ///< object
  std::unique_ptr<specfem::runtime_configuration::seismogram>
      seismogram; ///< Pointer to
                  ///< seismogram object
  std::unique_ptr<specfem::runtime_configuration::wavefield>
      wavefield; ///< Pointer to
                 ///< wavefield object
  std::unique_ptr<specfem::runtime_configuration::plot_wavefield>
      plot_wavefield; ///< Pointer to
                      ///< plot_wavefield object
  std::unique_ptr<specfem::runtime_configuration::kernel> kernel;
  std::unique_ptr<specfem::runtime_configuration::property> property;
  std::unique_ptr<specfem::runtime_configuration::database_configuration>
      databases; ///< Get database filenames
  std::unique_ptr<specfem::runtime_configuration::solver::solver>
      solver; ///< Pointer to solver object
};
} // namespace runtime_configuration
} // namespace specfem

#endif
