#include "../include/compute.h"
#include "../include/config.h"
#include "../include/domain.h"
#include "../include/kokkos_abstractions.h"
#include "../include/material.h"
#include "../include/mesh.h"
#include "../include/parameter_parser.h"
#include "../include/params.h"
#include "../include/read_mesh_database.h"
#include "../include/read_sources.h"
#include "../include/solver.h"
#include "../include/source.h"
#include "../include/specfem_mpi.h"
#include "../include/timescheme.h"
#include "../include/utils.h"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>
#include <stdexcept>
#include <string>
#include <vector>
// Specfem2d driver

boost::program_options::options_description define_args() {
  namespace po = boost::program_options;

  po::options_description desc{ "======================================\n"
                                "------------SPECFEM Kokkos------------\n"
                                "======================================" };

  desc.add_options()("help,h", "Print this help message")(
      "parameters_file,p", po::value<std::string>(),
      "Location to parameters file");

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

void execute(const std::string parameter_file, specfem::MPI::MPI *mpi) {

  specfem::runtime_configuration::setup setup(parameter_file);
  const auto [database_filename, source_filename] = setup.get_databases();

  mpi->cout(setup.print_header());

  // database_config database_config = get_node_config(database_file, mpi);

  // Set up GLL quadrature points
  auto [gllx, gllz] = setup.instantiate_quadrature();

  // Read mesh generated MESHFEM
  std::vector<specfem::material *> materials;
  specfem::mesh mesh(database_filename, materials, mpi);

  // Read sources
  //    if start time is not explicitly specified then t0 is determined using
  //    source frequencies and time shift
  auto [sources, t0] =
      specfem::read_sources(source_filename, setup.get_dt(), mpi);

  // Generate compute structs to be used by the solver
  specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                    gllz);
  specfem::compute::partial_derivatives partial_derivatives(
      mesh.coorg, mesh.material_ind.knods, gllx, gllz);
  specfem::compute::properties material_properties(mesh.material_ind.kmato,
                                                   materials, mesh.nspec,
                                                   gllx.get_N(), gllz.get_N());

  // Locate the sources
  for (auto &source : sources)
    source->locate(compute.coordinates.coord, compute.h_ibool, gllx.get_hxi(),
                   gllz.get_hxi(), mesh.nproc, mesh.coorg,
                   mesh.material_ind.knods, mesh.npgeo,
                   material_properties.h_ispec_type, mpi);

  // User output
  for (auto &source : sources) {
    if (mpi->main_proc())
      std::cout << *source << std::endl;
  }

  // Update solver intialization time
  setup.update_t0(-1.0 * t0);

  // Instantiate the solver and timescheme
  auto it = setup.instantiate_solver();

  // User output
  if (mpi->main_proc())
    std::cout << *it << std::endl;

  // Setup solver compute struct
  specfem::compute::sources compute_sources(sources, gllx, gllz, mpi);

  // Instantiate domain classes
  const int nglob = specfem::utilities::compute_nglob(compute.h_ibool);
  specfem::Domain::Domain *domains = new specfem::Domain::Elastic(
      ndim, nglob, &compute, &material_properties, &partial_derivatives,
      &compute_sources, &gllx, &gllz);

  specfem::solver::solver *solver =
      new specfem::solver::time_marching(domains, it);

  solver->run();
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
      execute(parameters_file, mpi);
    }
  }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
