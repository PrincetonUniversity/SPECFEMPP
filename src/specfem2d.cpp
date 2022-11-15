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
void equate(int computed_value, int ref_value) {
  if (computed_value != ref_value) {
    std::ostringstream ss;
    ss << "Computed value " << computed_value << " != ref value" << ref_value;

    throw std::runtime_error(ss.str());
  }
};
void equate(type_real computed_value, type_real ref_value) {
  if (fabs(computed_value - ref_value) >= 1e-4) {
    std::ostringstream ss;
    ss << "Computed value = " << computed_value
       << " != ref value = " << ref_value;

    throw std::runtime_error(ss.str());
  }
};

template <typename T>
void test_array(specfem::HostView3d<T> computed_array, std::string ref_file,
                int n1, int n2, int n3) {
  assert(computed_array.extent(0) == n1);
  assert(computed_array.extent(1) == n2);
  assert(computed_array.extent(2) == n3);

  T ref_value;
  std::ifstream stream;
  stream.open(ref_file);

  for (int i1 = 0; i1 < n1; i1++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i3 = 0; i3 < n3; i3++) {
        IO::fortran_IO::fortran_read_line(stream, &ref_value);
        try {
          equate(computed_array(i1, i2, i3), ref_value);
        } catch (std::runtime_error &e) {
          stream.close();
          std::ostringstream ss;
          ss << e.what() << ", at i1 = " << i1 << ", i2 = " << i2
             << ", i3 = " << i3;
          throw std::runtime_error(ss.str());
        }
      }
    }
  }
}

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

    specfem::compute::properties properties(mesh.material_ind.kmato, materials,
                                            mesh.nspec, gllz.get_N(),
                                            gllx.get_N());

    // std::string ref_file{ "/scratch/gpfs/rk9481/specfem2d_kokkos/tests/"
    //                       "unittests/compute/serial/data/xiz_00000.bin" };

    // test_array(coordinates.xiz, ref_file, mesh.nspec, gllz.get_N(),
    //            gllx.get_N());
  }

  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;

  return 0;
}
