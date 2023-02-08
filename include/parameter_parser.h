#ifndef PARAMETER_PARSER_H
#define PARAMETER_PARSER_H

#include "../include/config.h"
#include "../include/quadrature.h"
#include "../include/timescheme.h"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
/**
 * @brief Runtime configuration namespace defines objects needed to read and
 * instantiate the simulation
 *
 * Each object in runtime configuration is closely related to a node in
 * specfem_config.yaml used to setup a simulation
 */
namespace runtime_configuration {

/**
 * @brief Header class to store and print the title and description of the
 * simulation
 *
 */
class header {

public:
  /**
   * @brief Construct a new header object
   *
   * @param title Title of simulation
   * @param description Description of the simulation
   */
  header(std::string title, std::string description)
      : title(title), description(description){};
  /**
   * @brief Construct a new header object using YAML node
   *
   * @param Node YAML node as read from a specfem_config.yaml
   */
  header(const YAML::Node &Node);

  /**
   * @brief Get the title
   *
   * @return std::string title of the simulation
   */
  std::string get_title() { return this->title; }
  /**
   * @brief Get the description
   *
   * @return std::string description of the simulation
   */
  std::string get_description() { return this->description; }

  friend std::ostream &operator<<(std::ostream &out, header &header);

private:
  std::string title;       ///< Title of the simulation
  std::string description; ///< Description of the simulation
};

/**
 * @brief Quadrature object is used to read and instantiate the
 * specfem::quadrature::quadrature classes in different dimensions
 *
 */
class quadrature {
public:
  /**
   * @brief Construct a new quadrature object
   *
   * @param alpha alpha value used to instantiate a
   * specfem::quadrature::quadrature class
   * @param beta beta value used to instantiate a
   * specfem::quadrature::quadrature class
   * @param ngllx number of quadrature points in x-dimension
   * @param ngllz number of quadrature points in z-dimension
   */
  quadrature(type_real alpha, type_real beta, int ngllx, int ngllz)
      : alpha(alpha), beta(beta), ngllx(ngllx), ngllz(ngllz){};
  /**
   * @brief Construct a new quadrature object
   *
   * @param Node YAML node describing the quadrature as read from a
   * specfem_config.yaml
   */
  quadrature(const YAML::Node &Node);
  /**
   * @brief Instantiate quadrature objects in x and z dimensions
   *
   * @return std::tuple<specfem::quadrature::quadrature,
   * specfem::quadrature::quadrature> Quadrature objects in x and z dimensions
   */
  std::tuple<specfem::quadrature::quadrature, specfem::quadrature::quadrature>
  instantiate();

private:
  type_real alpha; ///< alpha value used to instantiate a
                   ///< specfem::quadrature::quadrature class
  type_real beta;  ///< beta value used to instantiate a
                   ///< specfem::quadrature::quadrature class
  int ngllx;       ///< number of quadrature points in x-dimension
  int ngllz;       ///< number of quadrature points in z-dimension
};

/**
 * @brief Solver class instantiates solver object which defines solution
 * algorithm for the Spectral Element Method
 *
 * @note Currently solver class is not implemented. Hence the solver class only
 * instantiates a timescheme object (In the future this class will only be
 * specific to time-marching (explicit) SEMs).
 *
 */
class solver {

public:
  /**
   * @brief Instantiate the Timescheme
   *
   * @return specfem::TimeScheme::TimeScheme* Pointer to the TimeScheme object
   * used in the solver algorithm
   */
  virtual specfem::TimeScheme::TimeScheme *instantiate();
  /**
   * @brief Update simulation start time.
   *
   * If user has not defined start time then we need to update the simulation
   * start time based on source frequencies and time shift
   *
   * @note This might be specific to only time-marching solvers
   *
   * @param t0 Simulation start time
   */
  virtual void update_t0(type_real t0){};
  /**
   * @brief Get the value of time increment
   *
   * @return type_real value of time increment
   */
  virtual type_real get_dt() const {
    std::runtime_error("Solver not instantiated properly");
    return 0.0;
  };
};

/**
 * @brief time_marching class is used to instantiate a time-marching solver
 *
 */
class time_marching : public solver {

public:
  /**
   * @brief Construct a new time marching object
   *
   * @param timescheme Type of timescheme
   * @param dt delta time of the timescheme
   * @param nstep Number of time steps
   */
  time_marching(std::string timescheme, type_real dt, type_real nstep)
      : timescheme(timescheme), dt(dt), nstep(nstep){};
  /**
   * @brief Construct a new time marching object
   *
   * @param Node YAML node describing the time-marching method as read from a
   * specfem_config.yaml
   */
  time_marching(const YAML::Node &Node);
  /**
   * @brief Update simulation start time.
   *
   * If user has not defined start time then we need to update the simulation
   * start time based on source frequencies and time shift
   *
   * @note This might be specific to only time-marching solvers
   *
   * @param t0 Simulation start time
   */
  void update_t0(type_real t0) override { this->t0 = t0; }
  /**
   * @brief Instantiate the Timescheme
   *
   * @return specfem::TimeScheme::TimeScheme* Pointer to the TimeScheme object
   * used in the solver algorithm
   */
  specfem::TimeScheme::TimeScheme *instantiate() override;
  /**
   * @brief Get the value of time increment
   *
   * @return type_real value of time increment
   */
  type_real get_dt() const override { return this->dt; }

private:
  int nstep;              ///< number of time steps
  type_real dt;           ///< delta time for the timescheme
  type_real t0;           ///< simulation start time
  std::string timescheme; ///< Time scheme e.g. Newmark, Runge-Kutta, LDDRK
};

/**
 * @brief Run setup defines run configuration for the simulation
 * @note This object is not used in the current version
 *
 */
class run_setup {

public:
  /**
   * @brief Construct a new run setup object
   *
   * @note This object is not used in the current version
   *
   * @param nproc Number of processors used in the simulation
   * @param nruns Number of simulation runs
   */
  run_setup(int nproc, int nruns) : nproc(nproc), nruns(nruns){};
  /**
   * @brief Construct a new run setup object
   *
   * @param Node YAML node describing the run configuration as read from a
   * specfem_config.yaml
   */
  run_setup(const YAML::Node &Node);

private:
  int nproc; ///< number of processors used in the simulation
  int nruns; ///< Number of simulation runs
};

/**
 * Setup class is used to read the specfem_config.yaml parameter file.
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
   * @param parameter_file specfem_config.yaml parameter file
   */
  setup(std::string parameter_file);
  /**
   * @brief Instantiate quadrature objects in x and z dimensions
   *
   * @return std::tuple<specfem::quadrature::quadrature,
   * specfem::quadrature::quadrature> Quadrature objects in x and z dimensions
   */
  std::tuple<specfem::quadrature::quadrature, specfem::quadrature::quadrature>
  instantiate_quadrature() {
    return this->quadrature->instantiate();
  }
  /**
   * @brief Instantiate the Timescheme
   *
   * @return specfem::TimeScheme::TimeScheme* Pointer to the TimeScheme object
   * used in the solver algorithm
   */
  specfem::TimeScheme::TimeScheme *instantiate_solver() {
    return this->solver->instantiate();
  }
  /**
   * @brief Update simulation start time.
   *
   * If user has not defined start time then we need to update the simulation
   * start time based on source frequencies and time shift
   *
   * @note This might be specific to only time-marching solvers
   *
   * @param t0 Simulation start time
   */
  void update_t0(type_real t0) { this->solver->update_t0(t0); }
  /**
   * @brief Log the header and description of the simulation
   */
  std::string print_header();

  type_real get_dt() const { return solver->get_dt(); }

private:
  specfem::runtime_configuration::header *header; ///< Pointer to header object
  specfem::runtime_configuration::solver *solver; ///< Pointer to solver object
  specfem::runtime_configuration::run_setup *run_setup;   ///< Pointer to
                                                          ///< run_setup object
  specfem::runtime_configuration::quadrature *quadrature; ///< Pointer to
                                                          ///< quadrature object
};

} // namespace runtime_configuration
} // namespace specfem

#endif
