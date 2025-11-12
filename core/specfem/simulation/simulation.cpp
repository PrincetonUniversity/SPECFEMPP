#include "specfem/simulation.hpp"
#include "context.hpp"
#include "enumerations/dimension.hpp"
#include "io/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "solver/solver.hpp"
#include "specfem/assembly.hpp"
#include "specfem/receivers.hpp"
#include "specfem/source.hpp"
#include "specfem/timescheme.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>

#include <sstream>

namespace {

std::string
print_end_message(std::chrono::time_point<std::chrono::system_clock> start_time,
                  std::chrono::duration<double> solver_time) {
  std::ostringstream message;
  // current date/time based on current system
  const auto now = std::chrono::system_clock::now();

  std::time_t c_now = std::chrono::system_clock::to_time_t(now);

  std::chrono::duration<double> diff = now - start_time;

  message << "Total simulation time : " << diff.count() << " secs\n"
          << "Total solver time (time loop) : " << solver_time.count()
          << " secs\n"
          << "Simulation end time : " << ctime(&c_now);

  return message.str();
}

// Internal function for 2D simulations
void simulation_2d(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    specfem::MPI::MPI *mpi) {

  // --- Read parameter file --------------------------------------
  auto start_time = std::chrono::system_clock::now();
  specfem::runtime_configuration::setup setup(parameter_dict, default_dict);

  // Set up logging (empty string means stdout)
  mpi->set_logfile(setup.get_output_log());

  // Print header
  specfem::simulation::model simulation_model =
      specfem::simulation::model::Cartesian2D;
  std::string title =
      "SPECFEM++ - " + specfem::simulation::to_string(simulation_model);
  mpi->print_header(title);

  const auto database_filename = setup.get_databases();
  const auto quadrature = setup.instantiate_quadrature();

  mpi->print(setup.print_header(start_time));

  // --- Read mesh and materials -----------------------------------
  mpi->print_title("Reading the mesh");
  const auto mesh = specfem::io::read_2d_mesh(
      database_filename, setup.get_elastic_wave_type(),
      setup.get_electromagnetic_wave_type(), mpi);

  // --- Read sources ----------------------------------------------
  mpi->print_title("Reading Sources");
  const int nsteps = setup.get_nsteps();
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  auto [sources, t0] =
      specfem::io::read_2d_sources(setup.get_sources(), nsteps, setup.get_t0(),
                                   setup.get_dt(), simulation_type);
  setup.update_t0(t0); // Update t0 in case it was changed

  mpi->print("Number of sources : " + std::to_string(sources.size()) + "\n");

  for (auto &source : sources) {
    mpi->print(source->print());
  }

  // --- Read receivers --------------------------------------------
  mpi->print_title("Reading Receivers");
  const auto stations_node = setup.get_stations();
  const auto angle = setup.get_receiver_angle();
  auto receivers = specfem::io::read_2d_receivers(stations_node, angle);

  mpi->print("Number of receivers : " + std::to_string(receivers.size()) +
             "\n");

  for (auto &receiver : receivers) {
    mpi->print(receiver->print());
  }

  // --- Generate Assembly -----------------------------------------
  mpi->print_title("Generating Assembly");
  const type_real dt = setup.get_dt();
  const int max_seismogram_time_step = setup.get_max_seismogram_step();
  const int nstep_between_samples = setup.get_nstep_between_samples();
  specfem::assembly::assembly<specfem::dimension::type::dim2> assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      nstep_between_samples, setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());

  mpi->print(assembly.print());

  // --- Instantiate Timescheme -----------------------------------
  mpi->print_title("Instantiating Timescheme");
  const auto time_scheme = setup.instantiate_timescheme(assembly.fields);
  mpi->print(time_scheme->print());

  // --- Instantiate Property writer -------------------------------
  const auto property_writer = setup.instantiate_property_writer();
  if (property_writer) {
    mpi->print_title("Writing model files");
    property_writer->write(assembly);
    return;
  }

  // --- Read wavefields ------------------------------------------
  const auto wavefield_reader = setup.instantiate_wavefield_reader();
  if (wavefield_reader) {
    mpi->print_title("Adding wavefield reader task");
    tasks.push_back(wavefield_reader);
  }

  // --- Instantiate Wavefield writer -----------------------------
  const auto wavefield_writer = setup.instantiate_wavefield_writer();
  if (wavefield_writer) {
    mpi->print_title("Adding wavefield writer task");
    tasks.push_back(wavefield_writer);
  }

  // --- Instantiate Wavefield plotter ----------------------------
  const auto wavefield_plotter =
      setup.instantiate_wavefield_plotter(assembly, dt, mpi);
  if (wavefield_plotter) {
    mpi->print_title("Adding wavefield plotter task");
    tasks.push_back(wavefield_plotter);
  }

  // --- Instantiate Solver ---------------------------------------
  mpi->print_title("Instantiating Solver");
  std::shared_ptr<specfem::solver::solver> solver =
      setup.instantiate_solver<5>(dt, assembly, time_scheme, tasks, mpi);

  // --- Execute Solver -------------------------------------------
  mpi->print_title("Executing time loop");
  const auto solver_start_time = std::chrono::system_clock::now();
  solver->run();
  const auto solver_end_time = std::chrono::system_clock::now();

  std::chrono::duration<double> solver_time =
      solver_end_time - solver_start_time;

  // --- Write Seismograms ----------------------------------------
  const auto seismogram_writer = setup.instantiate_seismogram_writer();
  if (seismogram_writer) {
    mpi->print_title("Writing seismogram files");
    seismogram_writer->write(assembly);
  }

  // --- Write Kernels --------------------------------------------
  const auto kernel_writer = setup.instantiate_kernel_writer();
  if (kernel_writer) {
    mpi->print_title("Writing kernel files");
    kernel_writer->write(assembly);
  }

  // --- Print end message ---------------------------------------
  mpi->print_title("Timing Summary");
  mpi->cout(print_end_message(start_time, solver_time));

  // Final message
  mpi->print_header("Finished simulation.");

  return;
}

// Internal function for 3D simulations
void simulation_3d(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    specfem::MPI::MPI *mpi) {

  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------
  auto start_time = std::chrono::system_clock::now();
  specfem::runtime_configuration::setup setup(parameter_dict, default_dict);
  const auto database_filename = setup.get_databases();
  const auto mesh_parameters_filename = setup.get_mesh_parameters();
  mpi->cout(setup.print_header(start_time));

  // Get simulation parameters
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------
  mpi->cout("Reading the mesh...");
  mpi->cout("===================");
  const auto quadrature = setup.instantiate_quadrature();
  const auto mesh = specfem::io::read_3d_mesh(mesh_parameters_filename,
                                              database_filename, mpi);
  std::chrono::duration<double> elapsed_seconds =
      std::chrono::system_clock::now() - start_time;
  mpi->cout("Time to read mesh: " + std::to_string(elapsed_seconds.count()) +
            " seconds");
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read Sources and Receivers
  // --------------------------------------------------------------
  auto [sources, t0] =
      specfem::io::read_3d_sources(setup.get_sources(), nsteps, setup.get_t0(),
                                   setup.get_dt(), simulation_type);
  setup.update_t0(t0); // Update t0 in case it was changed

  // TODO: Replace hardcoded receiver with proper 3D receiver reading
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::dimension::type::dim3> > >
      receivers;
  receivers.emplace_back(
      std::make_shared<
          specfem::receivers::receiver<specfem::dimension::type::dim3> >(
          "NET", "STA", 50000.0, 40000.0, 0.0));

  mpi->cout("Source Information:");
  mpi->cout("-------------------------------");
  if (mpi->main_proc()) {
    std::cout << "Number of sources : " << sources.size() << "\n" << std::endl;
  }

  for (auto &source : sources) {
    mpi->cout(source->print());
  }

  mpi->cout("Receiver Information:");
  mpi->cout("-------------------------------");

  if (mpi->main_proc()) {
    std::cout << "Number of receivers : " << receivers.size() << "\n"
              << std::endl;
  }

  for (auto &receiver : receivers) {
    mpi->cout(receiver->print());
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Generate Assembly
  // --------------------------------------------------------------
  const int max_seismogram_time_step = setup.get_max_seismogram_step();
  const int nstep_between_samples = setup.get_nstep_between_samples();
  specfem::assembly::assembly<specfem::dimension::type::dim3> assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      nstep_between_samples, setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());

  if (mpi->main_proc()) {
    mpi->cout(assembly.print());
  }

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Timescheme
  // --------------------------------------------------------------
  const auto time_scheme = setup.instantiate_timescheme(assembly.fields);

  if (mpi->main_proc())
    std::cout << *time_scheme << std::endl;

  if (mpi->main_proc())
    mpi->cout(assembly.print());

  // --------------------------------------------------------------
  // NOTE: Full 3D solver and writer support is not yet implemented
  // TODO: Implement the following for 3D:
  //   - Property writer
  //   - Wavefield reader/writer
  //   - Wavefield plotter
  //   - Solver instantiation and execution
  //   - Seismogram writer
  //   - Kernel writer
  // --------------------------------------------------------------

  return;
}

} // anonymous namespace

bool specfem::simulation::execute(
    const std::string &dimension, specfem::MPI::MPI *mpi,
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
        &tasks) {
  try {
    // Use simulation model enumeration for validation
    specfem::simulation::model simulation_model =
        specfem::simulation::from_string(dimension);

    switch (simulation_model) {
    case specfem::simulation::model::Cartesian2D:
      simulation_2d(parameter_dict, default_dict, tasks, mpi);
      return true;
    case specfem::simulation::model::Cartesian3D:
      simulation_3d(parameter_dict, default_dict, tasks, mpi);
      return true;
    default:
      std::cerr << "Unsupported simulation model" << std::endl;
      return false;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during execution: " << e.what() << std::endl;
    return false;
  }
}
