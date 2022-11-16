#ifndef READ_MATERIAL_PROPERTIES_H
#define READ_MATERIAL_PROPERTIES_H

#include "../include/material.h"
#include "../include/specfem_mpi.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace IO {

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
std::vector<specfem::material *>
read_material_properties(std::ifstream &stream, const int numat,
                         const specfem::MPI *mpi);
} // namespace IO

#endif
