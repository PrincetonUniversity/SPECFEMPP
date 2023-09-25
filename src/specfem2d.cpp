#include "compute/interface.hpp"
#include "coupled_interface/interface.hpp"
#include "domain/interface.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "receiver/interface.hpp"
#include "solver/interface.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>
// Specfem2d driver

std::string print_end_message(
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
    std::chrono::duration<double> solver_time) {
  std::ostringstream message;
  // current date/time based on current system
  const auto now = std::chrono::high_resolution_clock::now();

  std::time_t c_now = std::chrono::high_resolution_clock::to_time_t(now);

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

void execute(const std::string &parameter_file, const std::string &default_file,
             specfem::MPI::MPI *mpi) {

  // log start time
  auto start_time = std::chrono::high_resolution_clock::now();

  specfem::runtime_configuration::setup setup(parameter_file, default_file);
  const auto [database_filename, source_filename] = setup.get_databases();

  mpi->cout(setup.print_header(start_time));

  // Set up GLL quadrature points
  auto [gllx, gllz] = setup.instantiate_quadrature();

  // Read mesh generated MESHFEM
  std::vector<specfem::material::material *> materials;
  specfem::mesh::mesh mesh(database_filename, materials, mpi);

  // Read sources
  //    if start time is not explicitly specified then t0 is determined using
  //    source frequencies and time shift
  auto [sources, t0] =
      specfem::sources::read_sources(source_filename, setup.get_dt(), mpi);

  const auto stations_filename = setup.get_stations_file();
  const auto angle = setup.get_receiver_angle();
  auto receivers = specfem::receivers::read_receivers(stations_filename, angle);

  // Generate compute structs to be used by the solver
  specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                    gllz);
  specfem::compute::partial_derivatives partial_derivatives(
      mesh.coorg, mesh.material_ind.knods, gllx, gllz);
  specfem::compute::properties material_properties(
      mesh.material_ind.kmato, materials, mesh.nspec, gllx->get_N(),
      gllz->get_N());
  specfem::compute::coupled_interfaces::coupled_interfaces coupled_interfaces(
      compute.h_ibool, compute.coordinates.coord,
      material_properties.h_ispec_type, mesh.coupled_interfaces);

  // Print spectral element information
  mpi->cout(mesh.print(materials));

  // Locate the sources
  for (auto &source : sources)
    source->locate(compute.coordinates.coord, compute.h_ibool, gllx->get_hxi(),
                   gllz->get_hxi(), mesh.nproc, mesh.coorg,
                   mesh.material_ind.knods, mesh.npgeo,
                   material_properties.h_ispec_type, mpi);

  for (auto &receiver : receivers)
    receiver->locate(compute.coordinates.coord, compute.h_ibool,
                     gllx->get_hxi(), gllz->get_hxi(), mesh.nproc, mesh.coorg,
                     mesh.material_ind.knods, mesh.npgeo,
                     material_properties.h_ispec_type, mpi);

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

  // Update solver intialization time
  setup.update_t0(-1.0 * t0);

  // Instantiate the solver and timescheme
  auto it = setup.instantiate_solver();

  // Setup solver compute struct

  const type_real xmax = compute.coordinates.xmax;
  const type_real xmin = compute.coordinates.xmin;
  const type_real zmax = compute.coordinates.zmax;
  const type_real zmin = compute.coordinates.zmin;

  specfem::compute::sources compute_sources(sources, gllx, gllz, xmax, xmin,
                                            zmax, zmin, mpi);

  specfem::compute::receivers compute_receivers(
      receivers, setup.get_seismogram_types(), gllx, gllz, xmax, xmin, zmax,
      zmin, it->get_max_seismogram_step(), mpi);

  // Instantiate domain classes
  const int nglob = specfem::utilities::compute_nglob(compute.h_ibool);

  specfem::enums::element::quadrature::static_quadrature_points<5> qp5;
  specfem::domain::domain<
      specfem::enums::element::medium::acoustic,
      specfem::enums::element::quadrature::static_quadrature_points<5> >
      acoustic_domain_static(nglob, qp5, &compute, material_properties,
                             partial_derivatives, compute_sources,
                             compute_receivers, gllx, gllz);
  specfem::domain::domain<
      specfem::enums::element::medium::elastic,
      specfem::enums::element::quadrature::static_quadrature_points<5> >
      elastic_domain_static(nglob, qp5, &compute, material_properties,
                            partial_derivatives, compute_sources,
                            compute_receivers, gllx, gllz);

  // Instantiate coupled interfaces
  specfem::coupled_interface::coupled_interface acoustic_elastic_interface(
      acoustic_domain_static, elastic_domain_static, coupled_interfaces, qp5,
      partial_derivatives, compute.ibool, gllx->get_w(), gllz->get_w());

  specfem::coupled_interface::coupled_interface elastic_acoustic_interface(
      elastic_domain_static, acoustic_domain_static, coupled_interfaces, qp5,
      partial_derivatives, compute.ibool, gllx->get_w(), gllz->get_w());

  // Instantiate the writer
  auto writer =
      setup.instantiate_seismogram_writer(receivers, &compute_receivers);

  specfem::solver::solver *solver = new specfem::solver::time_marching<
      specfem::enums::element::quadrature::static_quadrature_points<5> >(
      acoustic_domain_static, elastic_domain_static, acoustic_elastic_interface,
      elastic_acoustic_interface, it);

  mpi->cout("Executing time loop:");
  mpi->cout("-------------------------------");

  const auto solver_start_time = std::chrono::high_resolution_clock::now();
  solver->run();
  const auto solver_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> solver_time =
      solver_end_time - solver_start_time;

  // Write only if a writer object has been defined
  if (writer) {
    mpi->cout("Writing seismogram files:");
    mpi->cout("-------------------------------");

    writer->write();
  }

  mpi->cout("Cleaning up:");
  mpi->cout("-------------------------------");

  for (auto &material : materials) {
    delete material;
  }

  for (auto &source : sources) {
    delete source;
  }

  for (auto &receiver : receivers) {
    delete receiver;
  }

  delete it;
  delete solver;
  delete writer;

  mpi->cout(print_end_message(start_time, solver_time));

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
      execute(parameters_file, default_file, mpi);
    }
  }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
