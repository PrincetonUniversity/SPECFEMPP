#ifndef PARAMETER_PARSER_H
#define PARAMETER_PARSER_H

#include "../include/config.h"
#include "../include/quadrature.h"
#include "../include/receiver.h"
#include "../include/timescheme.h"
#include "../include/writer.h"
#include "yaml-cpp/yaml.h"
#include <ctime>
#include <tuple>

namespace specfem {
/**
 * @brief Runtime configuration namespace defines objects needed to read and
 * instantiate the simulation
 *
 * Each object in runtime configuration is closely related to a node in
 * YAML file used to setup a simulation
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
   * @param Node YAML node as read from a YAML file
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
   * @param Node YAML node describing the quadrature
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
  virtual specfem::TimeScheme::TimeScheme *
  instantiate(const int nstep_between_samples);
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
    throw std::runtime_error("Solver not instantiated properly");
    return 0.0;
  };
  virtual type_real get_t0() const {
    throw std::runtime_error("Solver not instantiated properly");
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
   * @param Node YAML node describing the time-marching method
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
  specfem::TimeScheme::TimeScheme *
  instantiate(const int nstep_between_samples) override;
  /**
   * @brief Get the value of time increment
   *
   * @return type_real value of time increment
   */
  type_real get_dt() const override { return this->dt; }

  type_real get_t0() const override { return this->t0; }

private:
  int nstep;              ///< number of time steps
  type_real dt;           ///< delta time for the timescheme
  type_real t0;           ///< simulation start time
  std::string timescheme; ///< Time scheme e.g. Newmark, Runge-Kutta, LDDRK
};

/**
 * @brief Seismogram class is used to instantiate seismogram writer
 *
 */
class seismogram {

public:
  /**
   * @brief Construct a new seismogram object
   *
   * @param stations_file Path to stations file
   * @param angle Angle of the station
   * @param nstep_between_samples number of timesteps between seismogram
   * sampling (seismogram sampling frequency)
   * @param seismogram_type Type of seismogram
   * @param output_folder Path to folder location where seismogram will be
   * stored
   */
  seismogram(const std::string stations_file, const type_real angle,
             const int nstep_between_samples,
             const std::string seismogram_format,
             const std::string output_folder)
      : stations_file(stations_file), angle(angle),
        nstep_between_samples(nstep_between_samples),
        seismogram_format(seismogram_format), output_folder(output_folder){};
  /**
   * @brief Construct a new seismogram object
   *
   * @param Node YAML node describing the seismogram writer
   */
  seismogram(const YAML::Node &Node);
  /**
   * @brief Get the path of stations file
   *
   * @return std::string describing the locations of stations file
   */
  std::string get_stations_file() const { return this->stations_file; }
  /**
   * @brief Get the angle of the receiver
   *
   * @return type_real describing the angle of the receiver
   */
  type_real get_angle() const { return this->angle; };
  /**
   * @brief Get the number of time steps between seismogram sampling
   *
   * @return int descibing seismogram sampling frequency
   */
  int get_nstep_between_samples() const { return this->nstep_between_samples; }
  /**
   * @brief Get the types of seismogram requested
   *
   * @return std::vector<specfem::seismogram::type> vector seismogram types
   */
  std::vector<specfem::seismogram::type> get_seismogram_types() const {
    return stypes;
  }

  /**
   * @brief Instantiate a seismogram writer object
   *
   * @param receivers Vector of pointers to receiver objects used to instantiate
   * the writer
   * @param compute_receivers Pointer to specfem::compute::receivers struct used
   * to instantiate the writer
   * @param dt Time interval between timesteps
   * @param t0 Starting time of simulation
   * @return specfem::writer::writer* Pointer to an instantiated writer object
   */
  specfem::writer::writer *instantiate_seismogram_writer(
      std::vector<specfem::receivers::receiver *> &receivers,
      specfem::compute::receivers *compute_receivers, const type_real dt,
      const type_real t0) const;

private:
  std::string stations_file; ///< path to stations file
  type_real angle;           ///< Angle of the receiver
  int nstep_between_samples; ///< Seismogram sampling frequency
  std::vector<specfem::seismogram::type> stypes; ///< std::vector containing
                                                 ///< type of seismograms to be
                                                 ///< written
  std::string seismogram_format;                 ///< format of output file
  std::string output_folder;                     ///< Path to output folder
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
   * @param Node YAML node describing the run configuration
   */
  run_setup(const YAML::Node &Node);

private:
  int nproc; ///< number of processors used in the simulation
  int nruns; ///< Number of simulation runs
};

/**
 * @brief database_configuration defines the file location of databases
 *
 */
class database_configuration {

public:
  /**
   * @brief Construct a new database configuration object
   *
   * @param fortran_database location of fortran database
   * @param source_database location of source file
   */
  database_configuration(std::string fortran_database,
                         std::string source_database)
      : fortran_database(fortran_database), source_database(source_database){};
  /**
   * @brief Construct a new run setup object
   *
   * @param Node YAML node describing the run configuration
   */
  database_configuration(const YAML::Node &Node);

  std::tuple<std::string, std::string> get_databases() const {
    return std::make_tuple(this->fortran_database, this->source_database);
  }

private:
  std::string fortran_database; ///< location of fortran binary database
  std::string source_database;  ///< location of sources file
};

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
    auto it = this->solver->instantiate(
        this->seismogram->get_nstep_between_samples());

    // User output
    std::cout << *it << "\n";

    return it;
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
  std::string
  print_header(std::chrono::time_point<std::chrono::high_resolution_clock> now);

  /**
   * @brief Get delta time value
   *
   * @return type_real
   */
  type_real get_dt() const { return solver->get_dt(); }

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
    return seismogram->get_stations_file();
  }

  /**
   * @brief Get the angle of receivers
   *
   * @return type_real angle of the receiver
   */
  type_real get_receiver_angle() const { return seismogram->get_angle(); }

  /**
   * @brief Get the types of siesmograms to be calculated
   *
   * @return std::vector<specfem::seismogram::type> Types of seismograms to be
   * calculated
   */
  std::vector<specfem::seismogram::type> get_seismogram_types() const {
    return this->seismogram->get_seismogram_types();
  }

  /**
   * @brief Instantiate a seismogram writer object
   *
   * @param receivers Vector of pointers to receiver objects used to instantiate
   * the writer
   * @param compute_receivers Pointer to specfem::compute::receivers struct used
   * to instantiate the writer
   * @return specfem::writer::writer* Pointer to an instantiated writer object
   */
  specfem::writer::writer *instantiate_seismogram_writer(
      std::vector<specfem::receivers::receiver *> &receivers,
      specfem::compute::receivers *compute_receivers) const {
    return this->seismogram->instantiate_seismogram_writer(
        receivers, compute_receivers, this->solver->get_dt(),
        this->solver->get_t0());
  }

private:
  specfem::runtime_configuration::header *header; ///< Pointer to header object
  specfem::runtime_configuration::solver *solver; ///< Pointer to solver object
  specfem::runtime_configuration::run_setup *run_setup;   ///< Pointer to
                                                          ///< run_setup object
  specfem::runtime_configuration::quadrature *quadrature; ///< Pointer to
                                                          ///< quadrature object
  specfem::runtime_configuration::seismogram *seismogram; ///< Pointer to
                                                          ///< seismogram object
  specfem::runtime_configuration::database_configuration
      *databases; ///< Get database filenames
};

} // namespace runtime_configuration
} // namespace specfem

#endif
