#pragma once

#include "specfem/attenuation.hpp"
#include "specfem/datetime.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/receivers.hpp"
#include "specfem/runtime_configuration/sources.hpp"
#include "specfem/source.hpp"

#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include <optional>
#include <yaml-cpp/yaml.h>

// IO Modules
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/io/operators.hpp"
#include "specfem/io/reader.hpp"
#include "specfem/io/writer.hpp"

// Formats
#include "specfem/io_backends/ADIOS2/ADIOS2.hpp"
#include "specfem/io_backends/ASCII/ASCII.hpp"
#include "specfem/io_backends/HDF5/HDF5.hpp"
#include "specfem/io_backends/NPY/NPY.hpp"
#include "specfem/io_backends/NPZ/NPZ.hpp"

// Data Types
#include "specfem/io/kernel/writer.hpp"
#include "specfem/io/property/reader.hpp"
#include "specfem/io/property/writer.hpp"
#include "specfem/io/seismogram/reader.hpp"
#include "specfem/io/seismogram/writer.hpp"
#include "specfem/io/wavefield/reader.hpp"
#include "specfem/io/wavefield/writer.hpp"

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
 * @code
 * // Read 2D mesh from Fortran binary database file
 * auto mesh = specfem::io::read_2d_mesh(
 *      "DATABASES_MPI",
 *      specfem::enums::elastic_wave::psv,
 *      specfem::enums::electromagnetic_wave::te,
 *      specfem::attenuation::Setup{}); // attenuation disabled
 * @endcode
 *
 */
specfem::mesh::mesh<specfem::element::dimension_tag::dim2>
read_2d_mesh(const std::string &filename,
             const specfem::enums::elastic_wave wave,
             const specfem::enums::electromagnetic_wave electromagnetic_wave,
             const specfem::attenuation::Setup &attenuation_setup);

/**
 * @brief Construct a 3D mesh object from the SPECFEM++ 3D binary database.
 *
 * Reads boundary face data and classifies faces into absorbing boundaries.
 *
 * @param database_file Path to the binary mesh database
 * @param attenuation_setup Attenuation configuration
 * @return specfem::mesh::mesh<specfem::element::dimension_tag::dim3>
 *
 * @code
 * auto mesh = specfem::io::read_3d_mesh("database.bin",
 *                                        specfem::attenuation::Setup{});
 * @endcode
 */
specfem::mesh::mesh<specfem::element::dimension_tag::dim3>
read_3d_mesh(const std::string &database_file,
             const specfem::attenuation::Setup &attenuation_setup);

/** @brief Read a thin SPECFEM3D_GLOBE mesh database. */
specfem::mesh::mesh<specfem::element::dimension_tag::dim3>
read_globe_mesh(const std::string &database_file,
                const specfem::attenuation::Setup &attenuation_setup);

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
    specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>>
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
    specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>>
read_2d_receivers(const YAML::Node &stations, const type_real angle);

/**
 * @brief Read receivers file for 3D simulations
 *
 * Parse receiver stations file and create a vector of
 * specfem::receiver::receiver * object
 *
 * @param stations_file Stations file describing receiver locations
 * @param geographic If true, interpret the STATIONS columns as
 *   `name network latitude longitude elevation burial` (degrees/meters) and
 *   defer UTM projection to assembly time; if false (default), interpret them
 *   as `name network y x elevation z` cartesian coordinates (meters).
 * @return vector of instantiated receiver objects
 */
std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
read_3d_receivers(const std::string &stations_file, bool geographic = false);

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
 * @param geographic If true, interpret coordinates as geographic
 *   (latitude/longitude/depth); if false (default), as cartesian (x/y/z).
 * @return vector of instantiated receiver objects
 */
std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
read_3d_receivers(const YAML::Node &stations, bool geographic = false);

/**
 * @brief Read sources from parsed config entries (multi-format dispatch).
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param entries Source file entries with format and path
 * @param nsteps Number of time steps
 * @param user_t0 User defined t0
 * @param dt Time step
 * @param simulation_type Type of simulation
 * @return Tuple of (source objects, t0, optional start datetime)
 */
template <specfem::element::dimension_tag DimensionTag>
std::tuple<std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>,
           type_real, std::optional<specfem::datetime::type>>
read_sources(const std::vector<specfem::enums::source_file_entry> &entries,
             const int nsteps, const type_real user_t0, const type_real dt,
             const specfem::simulation::type simulation_type);

} // namespace io
} // namespace specfem
