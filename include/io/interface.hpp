#pragma once

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"
#include "specfem/receivers.hpp"
#include "specfem/source.hpp"

#include "specfem_setup.hpp"
#include <yaml-cpp/yaml.h>

namespace specfem {

/**
 * @namespace specfem::io
 * @brief Input/Output operations for SPECFEM++ simulations
 *
 * Provides high-level functions for reading mesh databases, sources, and
 * receivers, along with specialized readers and writers for wavefields,
 * seismograms, properties, and sensitivity kernels. Supports multiple I/O
 * backends (HDF5, ASCII, NPY, NPZ, ADIOS2).
 */
namespace io {

/**
 * @brief Construct a mesh object from a Fortran binary database file
 *
 * @param filename Fortran binary database filename
 * @return specfem::mesh::mesh Specfem mesh object for dimension type dim2
 *
 */
specfem::mesh::mesh<specfem::dimension::type::dim2>
read_2d_mesh(const std::string &filename,
             const specfem::enums::elastic_wave wave,
             const specfem::enums::electromagnetic_wave electromagnetic_wave);

/**
 * @brief Construct a 3D mesh object from MESHFEM3D Fortran binary database file
 *
 * @param database_file MESHFEM3D database file
 * @return specfem::mesh::mesh<specfem::dimension::type::dim3>
 *         Specfem mesh object for dimension type dim3
 *
 * @code
 * // Read 3D mesh from MESHFEM3D database file
 * auto mesh = specfem::io::read_3d_mesh("DATABASES_MPI");
 * @endcode
 */
specfem::mesh::mesh<specfem::dimension::type::dim3>
read_3d_mesh(const std::string &database_file);

/**
 * @brief Read station file
 *
 * Parse receiver stations file and create a vector of
 * specfem::receiver::receiver * object
 *
 * @param stations_file Stations file describing receiver locations
 * @param angle Angle of the receivers
 * @return vector of instantiated receiver objects
 */
std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::dimension::type::dim2> > >
read_2d_receivers(const std::string &stations_file, const type_real angle);

/**
 * @overload
 * @brief Read receivers from YAML Node
 *
 * Parse receiver stations file and create a vector of
 * specfem::receiver::receiver * object
 *
 * The receivers are defined in the YAML file as
 *
 * @code
 * receivers:
 *     stations-dict:
 *         - network: "network_name"
 *           station: "station_name"
 *           x: x_coordinate
 *           z: z_coordinate
 *         - <next station>
 * @endcode
 *
 * @param stations YAML node containing receiver locations
 * @param angle Angle of the receivers
 * @return vector of instantiated receiver objects
 */
std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::dimension::type::dim2> > >
read_2d_receivers(const YAML::Node &stations, const type_real angle);

/**
 * @brief Read receivers file for 3D simulations
 *
 * Parse receiver stations file and create a vector of
 * specfem::receiver::receiver * object
 *
 * @param stations_file Stations file describing receiver locations
 * @return vector of instantiated receiver objects
 */
std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::dimension::type::dim3> > >
read_3d_receivers(const std::string &stations_file);

/**
 * @overload
 * @brief Read 3D receivers from YAML Node
 *
 * Parse receiver stations file and create a vector of
 * specfem::receiver::receiver * object
 *
 * The receivers are defined in the YAML file as
 *
 * @code
 * receivers:
 *     stations-dict:
 *         - network: "network_name"
 *           station: "station_name"
 *           x: x_coordinate
 *           y: y_coordinate
 *           z: z_coordinate
 *         - <next station>
 * @endcode
 *
 * @param stations YAML node containing receiver locations
 * @return vector of instantiated receiver objects
 */
std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::dimension::type::dim3> > >
read_3d_receivers(const YAML::Node &stations);

/**
 * @brief Read sources file written in .yml format
 *
 * Parse source specification file written in yaml format and create a vector of
 * specfem::source::source * object
 *
 * @param sources_file Name of the yaml file
 * @param nsteps Number of time steps
 * @param user_t0 User defined t0
 * @param dt Time step
 * @param simulation_type Type of simulation
 *
 * @return std::vector<specfem::sources::source *> vector of instantiated source
 * objects
 */
std::tuple<std::vector<std::shared_ptr<
               specfem::sources::source<specfem::dimension::type::dim2> > >,
           type_real>
read_2d_sources(const std::string &sources_file, const int nsteps,
                const type_real user_t0, const type_real dt,
                const specfem::simulation::type simulation_type);

/**
 * @brief Read sources file written in .yml format
 *
 * Parse source specification file written in yaml format and create a vector of
 * specfem::source::source * object
 *
 * @param yaml YAML node containing source information
 * @param nsteps Number of time steps
 * @param user_t0 User defined t0
 * @param dt Time step
 * @param simulation_type Type of simulation
 * @return std::vector<specfem::sources::source *> vector of instantiated source
 * objects
 */
std::tuple<std::vector<std::shared_ptr<
               specfem::sources::source<specfem::dimension::type::dim2> > >,
           type_real>
read_2d_sources(const YAML::Node yaml, const int nsteps,
                const type_real user_t0, const type_real dt,
                const specfem::simulation::type simulation_type);

/**
 * @brief Read sources file written in .yml format
 *
 * Parse source specification file written in yaml format and create a vector of
 * specfem::source::source * object
 *
 * @param sources_file Name of the yaml file
 * @param nsteps Number of time steps
 * @param user_t0 User defined t0
 * @param dt Time step
 * @param simulation_type Type of simulation
 *
 * @return std::vector<specfem::sources::source *> vector of instantiated source
 * objects
 */
std::tuple<std::vector<std::shared_ptr<
               specfem::sources::source<specfem::dimension::type::dim3> > >,
           type_real>
read_3d_sources(const std::string &sources_file, const int nsteps,
                const type_real user_t0, const type_real dt,
                const specfem::simulation::type simulation_type);

/**
 * @brief Read sources file written in .yml format
 *
 * Parse source specification file written in yaml format and create a vector of
 * specfem::source::source * object
 *
 * @param yaml YAML node containing source information
 * @param nsteps Number of time steps
 * @param user_t0 User defined t0
 * @param dt Time step
 * @param simulation_type Type of simulation
 * @return std::vector<specfem::sources::source *> vector of instantiated source
 * objects
 */
std::tuple<std::vector<std::shared_ptr<
               specfem::sources::source<specfem::dimension::type::dim3> > >,
           type_real>
read_3d_sources(const YAML::Node yaml, const int nsteps,
                const type_real user_t0, const type_real dt,
                const specfem::simulation::type simulation_type);

} // namespace io
} // namespace specfem
