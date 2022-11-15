#include "../include/compute.h"
#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/material.h"
#include "../include/mesh.h"
#include "../include/params.h"
#include "../include/read_mesh_database.h"
#include "../include/specfem_mpi.h"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>
// Specfem2d driver

//-----------------------------------------------------------------
// config parser routines
struct config {
  std::string database_filename;
};

void operator>>(YAML::Node &Node, config &config) {
  config.database_filename = Node["database_file"].as<std::string>();
}

config get_node_config(std::string config_file, specfem::MPI *mpi) {
  // read specfem config file
  config config{};
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
      N >> config;
      return config;
    }
  }

  throw std::runtime_error("Could not process yaml file");

  // Dummy return type. Should never reach here.
  return config;
}
//-----------------------------------------------------------------

int main(int argc, char **argv) {

  // Initialize MPI
  specfem::MPI *mpi = new specfem::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  {

    std::string config_file = "../DATA/specfem_config.yaml";

    config config = get_node_config(config_file, mpi);

    // Set up GLL quadrature points
    quadrature::quadrature gllx(0.0, 0.0, ngll);
    quadrature::quadrature gllz(0.0, 0.0, ngll);

    specfem::parameters params;

    std::vector<specfem::material *> materials;
    specfem::mesh mesh(config.database_filename, materials, mpi);
    specfem::compute::coordinates coordinates(
        mesh.coorg, mesh.material_ind.knods, gllx, gllz);

    specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods,
                                      mesh.material_ind.kmato, gllx, gllz,
                                      materials);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;

  return 0;
}
