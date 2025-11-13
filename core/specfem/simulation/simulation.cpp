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

  message << "\n================================================\n"
          << "             Finished simulation\n"
          << "================================================\n\n"
          << "Total simulation time : " << diff.count() << " secs\n"
          << "Total solver time (time loop) : " << solver_time.count()
          << " secs\n"
          << "Simulation end time : " << ctime(&c_now)
          << "------------------------------------------------\n";

  return message.str();
}

// Internal function for 2D simulations
void simulation_2d(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    const std::shared_ptr<specfem::MPI::MPI> &mpi) {

  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------
  auto start_time = std::chrono::system_clock::now();
  specfem::runtime_configuration::setup setup(parameter_dict, default_dict);
  const auto database_filename = setup.get_databases();

  mpi->cout(setup.print_header(start_time));

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();
  const auto mesh = specfem::io::read_2d_mesh(
      database_filename, setup.get_elastic_wave_type(),
      setup.get_electromagnetic_wave_type(), *mpi);
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read Sources and Receivers
  // --------------------------------------------------------------
  const int nsteps = setup.get_nsteps();
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  auto [sources, t0] =
      specfem::io::read_2d_sources(setup.get_sources(), nsteps, setup.get_t0(),
                                   setup.get_dt(), simulation_type);
  setup.update_t0(t0); // Update t0 in case it was changed

  const auto stations_node = setup.get_stations();
  const auto angle = setup.get_receiver_angle();
  auto receivers = specfem::io::read_2d_receivers(stations_node, angle);

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
  const type_real dt = setup.get_dt();
  const int max_seismogram_time_step = setup.get_max_seismogram_step();
  const int nstep_between_samples = setup.get_nstep_between_samples();
  specfem::assembly::assembly<specfem::dimension::type::dim2> assembly(
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

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //               Write properties
  // --------------------------------------------------------------
  const auto property_writer = setup.instantiate_property_writer();
  if (property_writer) {
    mpi->cout("Writing model files:");
    mpi->cout("-------------------------------");

    property_writer->write(assembly);
    return;
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read wavefields
  // --------------------------------------------------------------
  const auto wavefield_reader = setup.instantiate_wavefield_reader();
  // if (wavefield_reader) {
  //   mpi->cout("Reading wavefield files:");
  //   mpi->cout("-------------------------------");

  //   wavefield_reader->read(assembly);
  //   // Transfer the buffer field to device
  //   assembly.fields.buffer.copy_to_device();
  // }
  if (wavefield_reader) {
    tasks.push_back(wavefield_reader);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                  Write Forward Wavefields
  // --------------------------------------------------------------
  const auto wavefield_writer = setup.instantiate_wavefield_writer();
  if (wavefield_writer) {
    tasks.push_back(wavefield_writer);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate plotter
  // --------------------------------------------------------------
  const auto wavefield_plotter =
      setup.instantiate_wavefield_plotter(assembly, dt, mpi);
  if (wavefield_plotter) {
    tasks.push_back(wavefield_plotter);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Solver
  // --------------------------------------------------------------
  std::shared_ptr<specfem::solver::solver> solver =
      setup.instantiate_solver<5>(dt, assembly, time_scheme, tasks);
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Execute Solver
  // --------------------------------------------------------------
  // Time the solver
  mpi->cout("Executing time loop:");
  mpi->cout("-------------------------------");

  const auto solver_start_time = std::chrono::system_clock::now();
  solver->run();
  const auto solver_end_time = std::chrono::system_clock::now();

  std::chrono::duration<double> solver_time =
      solver_end_time - solver_start_time;
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Write Seismograms
  // --------------------------------------------------------------
  const auto seismogram_writer = setup.instantiate_seismogram_writer();
  if (seismogram_writer) {
    mpi->cout("Writing seismogram files:");
    mpi->cout("-------------------------------");

    seismogram_writer->write(assembly);
  }
  // --------------------------------------------------------------

  // // --------------------------------------------------------------
  // //                  Write Forward Wavefields
  // // --------------------------------------------------------------
  // if (wavefield_writer) {
  //   mpi->cout("Writing wavefield files:");
  //   mpi->cout("-------------------------------");

  //   wavefield_writer->write(assembly);
  // }
  // // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                Write Kernels
  // --------------------------------------------------------------
  const auto kernel_writer = setup.instantiate_kernel_writer();
  if (kernel_writer) {
    mpi->cout("Writing kernel files:");
    mpi->cout("-------------------------------");

    kernel_writer->write(assembly);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Print End Message
  // --------------------------------------------------------------
  mpi->cout(print_end_message(start_time, solver_time));
  // --------------------------------------------------------------

  return;
}

// Internal function for 3D simulations
void simulation_3d(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    const std::shared_ptr<specfem::MPI::MPI> &mpi) {

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
                                              database_filename, *mpi);
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
    const std::string &dimension, std::weak_ptr<specfem::MPI::MPI> mpi_weak,
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
        &tasks) {
  try {
    // Lock the weak_ptr to get a shared_ptr
    auto mpi = mpi_weak.lock();
    if (!mpi) {
      std::cerr << "Error: MPI object is no longer valid" << std::endl;
      return false;
    }

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
