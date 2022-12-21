#ifndef READ_SOURCES_H
#define READ_SOURCES_H

#include "../include/config.h"
#include "../include/source.h"
#include <vector>

namespace specfem {
/**
 * @brief Read sources file written in .yml format
 *
 * Parse source specification file written in yaml format and create a vector of
 * specfem::source::source * object
 *
 * @param sources_file Name of the yml file
 * @param mpi Pointer to specfem MPI object
 * @return std::vector<specfem::sources::source *> vector of instantiated source
 * objects
 */
std::tuple<std::vector<specfem::sources::source *>, type_real>
read_sources(const std::string sources_file, const specfem::MPI::MPI *mpi);
} // namespace specfem

#endif
