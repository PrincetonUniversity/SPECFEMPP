#include "enumerations/dimension.hpp"
#include "io/interface.hpp"
#include "parameter_parser/interface.hpp"
#include "periodic_tasks/periodic_task.hpp"
#include "specfem/receivers.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <yaml-cpp/yaml.h>

boost::program_options::options_description define_args() {
  namespace po = boost::program_options;

  po::options_description desc{ "======================================\n"
                                "------------SPECFEM Kokkos------------\n"
                                "======================================" };

  desc.add_options()("help,h", "Print this help message")(
      "parameters_file,p", po::value<std::string>(),
      "Location to parameters file")(
      "default_file,d",
      po::value<std::string>()->default_value(__default_file__),
      "Location of default parameters file.");

  return desc;
}

int parse_args(int argc, char **argv,
               boost::program_options::variables_map &vm) {

  const auto desc = define_args();
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (!vm.count("parameters_file")) {
    std::cout << desc << std::endl;
    return 0;
  }

  return 1;
}

void execute(
    const YAML::Node parameter_dict, const YAML::Node default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    specfem::MPI::MPI *mpi) {

  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------
  auto start_time = std::chrono::system_clock::now();
  specfem::runtime_configuration::setup setup(parameter_dict, default_dict);
  const auto database_filename = setup.get_databases();
  const auto mesh_parameters_filename = setup.get_mesh_parameters();

  // Get simulation parameters
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------

  // Read mesh from the mesh database file
  mpi->cout("Reading the mesh...");
  mpi->cout("===================");
  const auto mesh = specfem::io::read_3d_mesh(mesh_parameters_filename,
                                              database_filename, mpi);
  std::chrono::duration<double> elapsed_seconds =
      std::chrono::system_clock::now() - start_time;
  mpi->cout("Time to read mesh: " + std::to_string(elapsed_seconds.count()) +
            " seconds");

  // --------------------------------------------------------------
  //                   Get Quadrature
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();

  // --------------------------------------------------------------
  //                   Get Sources
  // --------------------------------------------------------------
  auto [sources, t0] =
      specfem::io::read_3d_sources(setup.get_sources(), nsteps, setup.get_t0(),
                                   setup.get_dt(), simulation_type);
  setup.update_t0(0.0); // Update t0 in case it was changed

  mpi->cout("Source Information:");
  mpi->cout("-------------------------------");
  if (mpi->main_proc()) {
    std::cout << "Number of sources : " << sources.size() << "\n" << std::endl;
  }

  for (auto &source : sources) {
    mpi->cout(source->print());
  }

  // --------------------------------------------------------------
  //                   Get receivers
  // --------------------------------------------------------------
  // create single receiver receivers vector for now
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::dimension::type::dim3> > >
      receivers;
  receivers.emplace_back(
      std::make_shared<
          specfem::receivers::receiver<specfem::dimension::type::dim3> >(
          "NET", "STA", 50000.0, 40000.0, 0.0));

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
  //                   Instantiate Timescheme
  // --------------------------------------------------------------
  const auto time_scheme = setup.instantiate_timescheme();
  if (mpi->main_proc())
    std::cout << *time_scheme << std::endl;
  const int max_seismogram_time_step = time_scheme->get_max_seismogram_step();
  const int nstep_between_samples = time_scheme->get_nstep_between_samples();

  // --------------------------------------------------------------
  //                   Generate Assembly
  // --------------------------------------------------------------
  specfem::assembly::assembly<specfem::dimension::type::dim3> assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      nstep_between_samples, setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());
  if (mpi->main_proc())
    mpi->cout(assembly.print());

  return;
}

int main(int argc, char **argv) {
  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  {
    boost::program_options::variables_map vm;
    if (parse_args(argc, argv, vm)) {
      const std::string parameters_file =
          vm["parameters_file"].as<std::string>();
      const std::string default_file = vm["default_file"].as<std::string>();
      const YAML::Node parameter_dict = YAML::LoadFile(parameters_file);
      const YAML::Node default_dict = YAML::LoadFile(default_file);
      std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
          tasks;
      execute(parameter_dict, default_dict, tasks, mpi);
    }
  }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
