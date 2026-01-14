#include "specfem/program.hpp"
#include "context.hpp"
#include "enumerations/dimension.hpp"
#include "io/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh.hpp"
#include "specfem/receivers.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/solver.hpp"
#include "specfem/source.hpp"
#include "specfem/timescheme.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>

#include <sstream>

namespace specfem::program {

template <specfem::dimension::type DimensionTag>
std::string
print_header(const specfem::runtime_configuration::setup &setup,
             const std::chrono::time_point<std::chrono::system_clock> now) {

  std::ostringstream message;

  // convert now to string form
  const std::time_t c_now = std::chrono::system_clock::to_time_t(now);

  std::string dim;

  if constexpr (DimensionTag == specfem::dimension::type::dim2) {
    dim = "2D";
  } else if constexpr (DimensionTag == specfem::dimension::type::dim3) {
    dim = "3D";
  } else {
    throw std::runtime_error("Unsupported dimension for header print.");
  }

  message << "================================================\n"
          << "            SPECFEM++ " << dim << " SIMULATION\n"
          << "================================================\n\n"
          << "Title : " << setup.get_header().get_title() << "\n"
          << "Discription: " << setup.get_header().get_description() << "\n"
          << "Simulation start time: " << ctime(&c_now)
          << "------------------------------------------------\n";

  return message.str();
}

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
void program_2d(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::dimension::type::dim2> > >
        tasks) {

  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------
  auto start_time = std::chrono::system_clock::now();
  specfem::runtime_configuration::setup setup(parameter_dict, default_dict);
  const auto database_filename = setup.get_databases();

  specfem::Logger::info(
      print_header<specfem::dimension::type::dim2>(setup, start_time));

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();
  specfem::Logger::info("Quadrature:");
  specfem::Logger::info("-------------------------------");
  specfem::Logger::info(quadrature.to_string());

  const auto mesh = specfem::io::read_2d_mesh(
      database_filename, setup.get_elastic_wave_type(),
      setup.get_electromagnetic_wave_type());

  specfem::Logger::info("Mesh Information:");
  specfem::Logger::info("-------------------------------");
  specfem::Logger::info(mesh.print());

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

  specfem::Logger::info("Source Information:");
  specfem::Logger::info("-------------------------------");
  specfem::Logger::info(
      "Number of sources : " + std::to_string(sources.size()) + "\n");

  for (auto &source : sources) {
    specfem::Logger::info(source->print());
  }

  specfem::Logger::info("Receiver Information:");
  specfem::Logger::info("-------------------------------");
  specfem::Logger::info(
      "Number of receivers : " + std::to_string(receivers.size()) + "\n");

  for (auto &receiver : receivers) {
    specfem::Logger::info(receiver->print());
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

  specfem::Logger::info(assembly.print());

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Timescheme
  // --------------------------------------------------------------
  const auto time_scheme = setup.instantiate_timescheme(assembly.fields);
  specfem::Logger::info(time_scheme->to_string());
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //               Write properties
  // --------------------------------------------------------------
  const auto property_writer = setup.instantiate_property_writer();
  if (property_writer) {
    specfem::Logger::info("Writing model files:");
    specfem::Logger::info("-------------------------------");

    property_writer->write(assembly);
    return;
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read wavefields
  // --------------------------------------------------------------
  const auto wavefield_reader =
      setup.instantiate_wavefield_reader<specfem::dimension::type::dim2>();
  if (wavefield_reader) {
    tasks.push_back(wavefield_reader);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                  Write Forward Wavefields
  // --------------------------------------------------------------
  const auto wavefield_writer =
      setup.instantiate_wavefield_writer<specfem::dimension::type::dim2>();
  if (wavefield_writer) {
    tasks.push_back(wavefield_writer);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate plotter
  // --------------------------------------------------------------
  const auto wavefield_plotter =
      setup.instantiate_wavefield_plotter(assembly, dt);
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
  specfem::Logger::info("Executing time loop:");
  specfem::Logger::info("-------------------------------");

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
    specfem::Logger::info("Writing seismogram files:");
    specfem::Logger::info("-------------------------------");

    seismogram_writer->write(assembly);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                Write Kernels
  // --------------------------------------------------------------
  const auto kernel_writer = setup.instantiate_kernel_writer();
  if (kernel_writer) {
    specfem::Logger::info("Writing kernel files:");
    specfem::Logger::info("-------------------------------");

    kernel_writer->write(assembly);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Print End Message
  // --------------------------------------------------------------
  specfem::Logger::info(print_end_message(start_time, solver_time));
  // --------------------------------------------------------------

  return;
}

// Internal function for 3D simulations
void program_3d(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::dimension::type::dim3> > >
        tasks) {

  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------
  auto start_time = std::chrono::system_clock::now();
  specfem::runtime_configuration::setup setup(parameter_dict, default_dict);
  const auto database_filename = setup.get_databases();

  specfem::Logger::info(
      print_header<specfem::dimension::type::dim3>(setup, start_time));

  // Get simulation parameters
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------
  specfem::Logger::info("Reading the mesh...");
  specfem::Logger::info("===================");
  auto mesh_start_time = std::chrono::system_clock::now();
  const auto mesh = specfem::io::read_3d_mesh(database_filename);
  auto mesh_read_time = std::chrono::system_clock::now() - mesh_start_time;
  specfem::Logger::info("Time to read mesh: " +
                        std::to_string(mesh_read_time.count()) + " seconds");
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Quadrature
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();
  specfem::Logger::info("Quadrature:");
  specfem::Logger::info("-------------------------------");
  specfem::Logger::info(quadrature.to_string());
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Get Sources
  // --------------------------------------------------------------
  auto [sources, t0] =
      specfem::io::read_3d_sources(setup.get_sources(), nsteps, setup.get_t0(),
                                   setup.get_dt(), simulation_type);
  setup.update_t0(t0); // Update t0 in case it was changed

  specfem::Logger::info("Source Information:");
  specfem::Logger::info("---------------------");
  specfem::Logger::info(
      "Number of sources : " + std::to_string(sources.size()) + "\n");

  for (auto &source : sources) {
    specfem::Logger::info(source->print());
  }

  // --------------------------------------------------------------
  //                   Get receivers
  // --------------------------------------------------------------
  // create single receiver receivers vector for now
  auto receivers = specfem::io::read_3d_receivers(setup.get_stations());

  specfem::Logger::info("Receiver Information:");
  specfem::Logger::info("---------------------");
  specfem::Logger::info(
      "Number of receivers : " + std::to_string(receivers.size()) + "\n");

  for (auto &receiver : receivers) {
    specfem::Logger::info(receiver->print());
  }

  // --------------------------------------------------------------
  //                   Generate Assembly
  // --------------------------------------------------------------
  specfem::Logger::info("Generating Assembly:");
  specfem::Logger::info("-------------------------------");
  const int max_seismogram_time_step = setup.get_max_seismogram_step();
  const int nstep_between_samples = setup.get_nstep_between_samples();
  specfem::assembly::assembly<specfem::dimension::type::dim3> assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      nstep_between_samples, setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());

  specfem::Logger::info(assembly.print());

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Timescheme
  // --------------------------------------------------------------
  const auto time_scheme = setup.instantiate_timescheme(assembly.fields);
  specfem::Logger::info(time_scheme->to_string());
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate plotter
  // --------------------------------------------------------------
  specfem::Logger::info("(If set) Instantiate wavefield plotter");
  specfem::Logger::info("-------------------------------");
  const auto wavefield_plotter =
      setup.instantiate_wavefield_plotter(assembly, dt);
  if (wavefield_plotter) {
    tasks.push_back(wavefield_plotter);
  }

  // --------------------------------------------------------------
  //                   Instantiate Solver
  // --------------------------------------------------------------
  specfem::Logger::info("Instantiate solver");
  specfem::Logger::info("-------------------------------");
  std::shared_ptr<specfem::solver::solver> solver =
      setup.instantiate_solver<5>(dt, assembly, time_scheme, tasks);
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Execute Solver
  // --------------------------------------------------------------
  // Time the solver
  specfem::Logger::info("Executing time loop:");
  specfem::Logger::info("-------------------------------");

  const auto solver_start_time = std::chrono::system_clock::now();
  solver->run();
  const auto solver_end_time = std::chrono::system_clock::now();

  std::chrono::duration<double> solver_time =
      solver_end_time - solver_start_time;

  specfem::Logger::info("Solver time: " + std::to_string(solver_time.count()) +
                        " seconds.");
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Write Seismograms
  // --------------------------------------------------------------
  const auto seismogram_writer = setup.instantiate_seismogram_writer();
  if (seismogram_writer) {
    specfem::Logger::info("Writing seismogram files.");
    seismogram_writer->write(assembly);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Print End Message
  // --------------------------------------------------------------
  specfem::Logger::info(print_end_message(start_time, solver_time));
  // --------------------------------------------------------------

  return;
}

bool execute(const std::string &dimension, const YAML::Node &parameter_dict,
             const YAML::Node &default_dict) {
  try {
    // Use simulation model enumeration for validation
    specfem::simulation::model simulation_model =
        specfem::simulation::from_string(dimension);

    switch (simulation_model) {
    case specfem::simulation::model::Cartesian2D: {
      // Setup periodic tasks (signal checking)
      const auto dimension_tag = specfem::dimension::type::dim2;
      std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<dimension_tag> > >
          tasks;
      const auto signal_task = std::make_shared<
          specfem::periodic_tasks::check_signal<dimension_tag> >(10);
      tasks.push_back(signal_task);

      // Run 2D Cartesian program
      program_2d(parameter_dict, default_dict, tasks);

      return true;
    }
    case specfem::simulation::model::Cartesian3D: {
      // Setup periodic tasks (signal checking)
      const auto dimension_tag = specfem::dimension::type::dim3;
      std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<dimension_tag> > >
          tasks;
      const auto signal_task = std::make_shared<
          specfem::periodic_tasks::check_signal<dimension_tag> >(10);
      tasks.push_back(signal_task);

      // Run 3D Cartesian program
      program_3d(parameter_dict, default_dict, tasks);

      return true;
    }
    default: {
      specfem::Logger::error("Unsupported simulation model");
      return false;
    }
    }
  } catch (const std::exception &e) {
    specfem::Logger::error(std::string("Error during execution: ") + e.what());
    return false;
  }
}

} // namespace specfem::program
