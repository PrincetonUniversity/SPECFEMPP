#ifndef _READ_SOURCES_HPP
#define _READ_SOURCES_HPP

#include "source.hpp"
#include <memory>

namespace specfem {
namespace sources {
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
std::tuple<std::vector<std::shared_ptr<specfem::sources::source> >, type_real>
read_sources(const std::string sources_file, const type_real dt);
} // namespace sources
} // namespace specfem

#endif
