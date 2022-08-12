#include "../include/fortran_IO.h"
#include "../include/mesh.h"
#include "../include/params.h"
#include "../include/specfem_mpi.h"
#include <fstream>
#include <iostream>

namespace IO {
void read_mesh_database_header(std::ifstream &stream, specfem::mesh &mesh,
                               specfem::MPI *mpi);
void read_coorg_elements(std::ifstream &stream, specfem::mesh &mesh,
                         specfem::MPI *mpi);
void read_mesh_database_attenuation(std::ifstream &stream,
                                    specfem::parameters &params,
                                    specfem::MPI *mpi);

} // namespace IO
