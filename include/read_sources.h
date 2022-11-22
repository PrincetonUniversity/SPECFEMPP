#ifndef READ_SOURCES_H
#define READ_SOURCES_H

#include "../include/config.h"
#include "../include/source.h"
#include <vector>

namespace specfem {
std::vector<specfem::sources::source *>
read_sources(const std::string sources_file, const specfem::MPI::MPI *mpi)
}

#endif
