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
#include <stdexcept>
#include <string>
#include <vector>
// Specfem2d driver

//-----------------------------------------------------------------
// config parser routines
struct database_config {
  std::string database_filename, source_filename;
};

void operator>>(YAML::Node &Node, database_config &database_config) {
  database_config.database_filename = Node["database_file"].as<std::string>();
  database_config.source_filename = Node["source_file"].as<std::string>();
}

database_config get_node_config(std::string config_file,
                                specfem::MPI::MPI *mpi) {
  // read specfem config file
  database_config database_config{};
  YAML::Node yaml = YAML::LoadFile(config_file);
  YAML::Node Node = yaml["databases"];
  assert(Node.IsSequence());
  if (Node.size() != mpi->get_size()) {
    std::ostringstream message;
    message << "Specfem configuration file generated with " << Node.size()
            << " number of processors. Current run is with nproc = "
            << mpi->get_size()
            << " Please run the code with nprocs = " << Node.size();
    throw std::runtime_error(message.str());
  }
  for (auto N : Node) {
    if (N["processor"].as<int>() == mpi->get_rank()) {
      N >> database_config;
      return database_config;
    }
  }

  throw std::runtime_error("Could not process yaml file");

  // Dummy return type. Should never reach here.
  return database_config;
}
//-----------------------------------------------------------------

int main(int argc, char **argv) {

  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  {

    std::string parameter_file = "../DATA/specfem_config.yaml";
    std::string database_file = "../DATA/databases.yaml";
    specfem::runtime_configuration::setup setup(parameter_file);

    mpi->cout(setup.print_header());

    database_config database_config = get_node_config(database_file, mpi);

    // Set up GLL quadrature points
    auto [gllx, gllz] = setup.instantiate_quadrature();

    // Read mesh generated MESHFEM
    std::vector<specfem::material *> materials;
    specfem::mesh mesh(database_config.database_filename, materials, mpi);

    // Read sources
    //    if start time is not explicitly specified then t0 is determined using
    //    source frequencies and time shift
    auto [sources, t0] = specfem::read_sources(database_config.source_filename,
                                               setup.get_dt(), mpi);

    // Generate compute structs to be used by the solver
    specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                      gllz);
    specfem::compute::partial_derivatives partial_derivatives(
        mesh.coorg, mesh.material_ind.knods, gllx, gllz);
    specfem::compute::properties material_properties(
        mesh.material_ind.kmato, materials, mesh.nspec, gllx.get_N(),
        gllz.get_N());

    // Locate the sources
    for (auto &source : sources)
      source->locate(compute.ibool, compute.coordinates.coord, gllx.get_hxi(),
                     gllz.get_hxi(), mesh.nproc, mesh.coorg,
                     mesh.material_ind.knods, mesh.npgeo,
                     material_properties.ispec_type, mpi);

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
    const int nglob = specfem::utilities::compute_nglob(compute.ibool);
    specfem::Domain::Domain *domains = new specfem::Domain::Elastic(
        ndim, nglob, &compute, &material_properties, &partial_derivatives,
        &compute_sources, &gllx, &gllz);

    specfem::solver::solver *solver =
        new specfem::solver::time_marching(domains, it);

    solver->run();
  }

  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;

  return 0;
}
