#ifndef _READ_SOURCES_HPP
#define _READ_SOURCES_HPP

#include "enumerations/simulation.hpp"
#include "source/interface.hpp"
#include <memory>

namespace specfem {
namespace IO {
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
read_sources(const std::string sources_file, const int nsteps,
             const type_real user_t0, const type_real dt,
             const specfem::simulation::type simulation_type);

} // namespace IO
} // namespace specfem

#endif
