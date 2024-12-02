#pragma once

#include "enumerations/simulation.hpp"
#include "mesh/mesh.hpp"
#include "receiver/interface.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"

namespace specfem {

namespace IO {

/**
 * @brief Construct a mesh object from a Fortran binary database file
 *
 * @param filename Fortran binary database filename
 * @param mpi pointer to MPI object to manage communication
 * @return specfem::mesh::mesh Specfem mesh object
 *
 */
specfem::mesh::mesh read_mesh(const std::string filename,
                              const specfem::MPI::MPI *mpi);

/**
 * @brief Read receiver station file
 *
 * Parse receiver stations file and create a vector of
 * specfem::source::source * object
 *
 * @param stations_file Stations file describing receiver locations
 * @param angle Angle of the receivers
 * @return std::vector<specfem::receivers::receiver *> vector of instantiated
 * receiver objects
 */
std::vector<std::shared_ptr<specfem::receivers::receiver> >
read_receivers(const std::string stations_file, const type_real angle);

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
