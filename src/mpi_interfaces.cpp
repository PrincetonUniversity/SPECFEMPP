#include "../include/mpi_interfaces.h"
#include "../include/kokkos_abstractions.h"

specfem::interfaces::interface::interface(const int ninterfaces,
                                          const int max_interface_size) {

  this->ninterfaces = ninterfaces;
  this->max_interface_size = max_interface_size;

#ifdef MPI_PARALLEL
  if (ninterfaces > 0) {
    this->my_neighbors = specfem::HostView1d<int>(
        "specfem::mesh::interfaces::my_neighbors", ninterfaces);
    this->my_nelmnts_neighbors = specfem::HostView1d<int>(
        "specfem::mesh::interfaces::my_nelmnts_neighbors", ninterfaces);
    this->my_interfaces =
        specfem::HostView3d<int>("specfem::mesh::interfaces::my_interfaces",
                                 ninterfaces, max_interface_size, 4);

    // initialize values
    for (int i = 0; i < ninterfaces; i++) {
      this->my_neighbors(i) = -1;
      this->my_nelmnts_neighbors(i) = 0;
      for (int j = 0; j < max_interface_size; j++) {
        for (int k = 0; k < 4; k++) {
          this->my_interfaces(i, j, k) = -1;
        }
      }
    }
  } else {
    this->my_neighbors =
        specfem::HostView1d<int>("specfem::mesh::interfaces::my_neighbors", 1);
    this->my_nelmnts_neighbors = specfem::HostView1d<int>(
        "specfem::mesh::interfaces::my_nelmnts_neighbors", 1);
    this->my_interfaces = specfem::HostView3d<int>(
        "specfem::mesh::interfaces::my_interfaces", 1, 1, 1);

    // initialize values
    this->my_neighbors(1) = -1;
    this->my_nelmnts_neighbors(1) = 0;
    this->my_interfaces(1, 1, 1) = -1;
  }
#else
  if (ninterfaces > 0)
    throw std::runtime_error("Found interfaces but SPECFEM compiled without "
                             "MPI. Compile SPECFEM with MPI");
  this->my_neighbors =
      specfem::HostView1d<int>("specfem::mesh::interfaces::my_neighbors", 1);
  this->my_nelmnts_neighbors = specfem::HostView1d<int>(
      "specfem::mesh::interfaces::my_nelmnts_neighbors", 1);
  this->my_interfaces = specfem::HostView3d<int>(
      "specfem::mesh::interfaces::my_interfaces", 1, 1, 1);

  // initialize values
  this->my_neighbors(1) = -1;
  this->my_nelmnts_neighbors(1) = 0;
  this->my_interfaces(1, 1, 1) = -1;
#endif

  return;
}
