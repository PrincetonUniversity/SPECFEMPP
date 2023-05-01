#ifndef _READ_MATERIAL_PROPERTIES_HPP
#define _READ_MATERIAL_PROPERTIES_HPP

#include "material/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include <fstream>
#include <iostream>
#include <vector>

namespace specfem {
namespace mesh {
namespace IO {
namespace fortran {

/**
 * Read material properties from a fotran binary database
 *
 * @param stream Stream object for fortran binary file buffered to materials
 * section
 * @param numat Number of materials to be read
 * @param mpi Pointer to MPI object
 * @return std::vector<specfem::material *> Pointer to material objects read
 * from the database file
 */
std::vector<specfem::material::material *>
read_material_properties(std::ifstream &stream, const int numat,
                         const specfem::MPI::MPI *mpi);
} // namespace fortran
} // namespace IO
} // namespace mesh
} // namespace specfem
#endif
