#ifndef READ_MATERIAL_PROPERTIES_H
#define READ_MATERIAL_PROPERTIES_H

#include "../include/material.h"
#include "../include/specfem_mpi.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace IO {

std::vector<specfem::material>
read_material_properties(std::ifstream &stream, int numat, specfem::MPI *mpi);
}

#endif
