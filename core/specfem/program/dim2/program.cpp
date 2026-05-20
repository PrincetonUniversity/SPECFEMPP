#include "program.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/io.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh.hpp"
#include "specfem/receivers.hpp"
#include "specfem/solver.hpp"
#include "specfem/source.hpp"
#include "specfem/timescheme.hpp"

namespace specfem::program {

void program_2d(
    const YAML::Node &parameter_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::element::dimension_tag::dim2>>>
        tasks) {

  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------
  auto start_time = std::chrono::system_clock::now();
  specfem::runtime_configuration::setup setup(parameter_dict);
  const auto database_filename = setup.get_databases();

  specfem::Logger::info([&](std::ostringstream &oss) {
    oss << print_header<specfem::element::dimension_tag::dim2>(setup,
                                                               start_time);
  });

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();
  specfem::Logger::info([&](std::ostringstream &oss) {
    oss << "Quadrature:\n"
        << "-------------------------------\n"
        << quadrature.to_string();
  });

  const auto mesh = specfem::io::read_2d_mesh(
      database_filename, setup.get_elastic_wave_type(),
      setup.get_electromagnetic_wave_type(), setup.get_attenuation_setup());

  // mesh.print() always called (not wrapped in lambda function) because it
  // requires collective communication
  const auto mesh_info = mesh.print();
  specfem::Logger::info([&](std::ostringstream &oss) {
    oss << "Mesh Information:\n"
        << "-------------------------------\n"
        << mesh_info;
  });

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read Sources and Receivers
  // --------------------------------------------------------------
  const int nsteps = setup.get_nsteps();
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  auto [sources, t0] =
      specfem::io::read_sources<specfem::element::dimension_tag::dim2>(
          setup.get_source_entries(), nsteps, setup.get_t0(), setup.get_dt(),
          simulation_type);
  setup.update_t0(t0); // Update t0 in case it was changed

  const auto stations_node = setup.get_stations();
  const auto angle = setup.get_receiver_angle();
  auto receivers = specfem::io::read_2d_receivers(stations_node, angle);
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Generate Assembly
  // --------------------------------------------------------------
  const type_real dt = setup.get_dt();
  const int max_seismogram_time_step = setup.get_max_seismogram_step();
  const int nstep_between_samples = setup.get_nstep_between_samples();
  specfem::assembly::assembly<specfem::element::dimension_tag::dim2> assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      nstep_between_samples, setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader(),
      setup.get_flux_scheme_configuration());

  specfem::Logger::info([&](std::ostringstream &oss) {
    oss << "Source Information:\n"
        << "-------------------------------\n"
        << "Number of sources : " << sources.size() << "\n";
    for (auto &source : sources) {
      oss << source->print();
    }

    oss << "Receiver Information:\n"
        << "-------------------------------\n"
        << "Number of receivers : " << receivers.size() << "\n";
    for (auto &receiver : receivers) {
      oss << receiver->print();
    }
  });

  // assembly.print() always called (not wrapped in lambda function)
  // because it requires collective communication
  specfem::Logger::info(assembly.print());

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Timescheme
  // --------------------------------------------------------------
  const auto time_scheme = setup.instantiate_timescheme(assembly.fields);
  specfem::Logger::info(
      [&](std::ostringstream &oss) { oss << time_scheme->to_string(); });
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //               Write properties
  // --------------------------------------------------------------
  const auto property_writer = setup.instantiate_property_writer();
  if (property_writer) {
    specfem::Logger::info(
        "Writing model files:\n-------------------------------");

    property_writer->write(assembly);
    return;
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read wavefields
  // --------------------------------------------------------------
  const auto wavefield_reader = setup.instantiate_wavefield_reader<
      specfem::element::dimension_tag::dim2>();
  if (wavefield_reader) {
    tasks.push_back(wavefield_reader);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                  Write Forward Wavefields
  // --------------------------------------------------------------
  const auto wavefield_writer = setup.instantiate_wavefield_writer<
      specfem::element::dimension_tag::dim2>();
  if (wavefield_writer) {
    tasks.push_back(wavefield_writer);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate plotter
  // --------------------------------------------------------------
  const auto wavefield_plotter = setup.instantiate_wavefield_plotter(assembly);
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
  specfem::Logger::info(
      "Executing time loop:\n-------------------------------");

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
    specfem::Logger::info(
        "Writing seismogram files:\n-------------------------------");

    seismogram_writer->write(assembly);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                Write Kernels
  // --------------------------------------------------------------
  const auto kernel_writer = setup.instantiate_kernel_writer();
  if (kernel_writer) {
    specfem::Logger::info(
        "Writing kernel files:\n-------------------------------");

    kernel_writer->write(assembly);
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Print End Message
  // --------------------------------------------------------------
  specfem::Logger::info([&](std::ostringstream &oss) {
    oss << print_end_message(start_time, solver_time);
  });
  // --------------------------------------------------------------

  return;
}

} // namespace specfem::program
