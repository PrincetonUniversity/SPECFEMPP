#include "../include/mpi_interfaces.h"
#include "../include/fortran_IO.h"
#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"
#include <tuple>

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

specfem::interfaces::interface::interface(std::ifstream &stream,
                                          const specfem::MPI::MPI *mpi) {

  // read number of interfaces
  // Where these 2 values are written needs to change in new database format
  int ninterfaces, max_interface_size;
  IO::fortran_IO::fortran_read_line(stream, &ninterfaces, &max_interface_size);

  mpi->cout("Number of interaces = " + std::to_string(ninterfaces));

  // allocate interface variables
  *this = specfem::interfaces::interface(ninterfaces, max_interface_size);

  // note: for serial simulations, ninterface will be zero.
  //       thus no further reading will be done below

  // reads in interfaces
#ifdef MPI_PARALLEL
  for (int num_interface = 0; num_interface < this->ninterfaces;
       num_interface++) {
    // format: #process_interface_id  #number_of_elements_on_interface
    // where
    //     process_interface_id = rank of (neighbor) process to share MPI
    //     interface with number_of_elements_on_interface = number of interface
    //     elements
    IO::fortran_IO::fortran_read_line(
        stream, &this->my_neighbors(num_interface),
        &this->my_nelmnts_neighbors(num_interface));
    // loops over interface elements
    for (int ie = 0; ie < this->my_nelmnts_neighbors(num_interface); ie++) {
      //   format: #(1)spectral_element_id  #(2)interface_type  #(3)node_id1
      //   #(4)node_id2

      //   interface types:
      //       1  -  corner point only
      //       2  -  element edge
      IO::fortran_IO::fortran_read_line(
          stream, &this->my_interfaces(num_interface, ie, 0),
          &this->my_interfaces(num_interface, ie, 1),
          &this->my_interfaces(num_interface, ie, 2),
          &this->my_interfaces(num_interface, ie, 3));
    }
  }
#endif

  return;
}
