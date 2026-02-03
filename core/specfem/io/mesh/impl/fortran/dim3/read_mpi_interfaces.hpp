#pragma once

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3 {

/**
 * @brief Read MPI interface data from MESHFEM3D database file
 *
 * Reads MPI interface information from a binary MESHFEM3D database file
 * for parallel domain decomposition in 3D spectral element simulations.
 * MPI interfaces define the communication patterns between neighboring
 * MPI processes in distributed memory parallel computations.
 *
 * @param stream Input file stream positioned at the MPI interfaces section of
 * the MESHFEM3D database
 *
 * @throws std::ios_base::failure If stream operations encounter errors
 *
 * @note **Current Status**: This function is a placeholder implementation that
 * validates the presence of MPI interface data but does not process it. Full
 * MPI interface support for 3D simulations is planned for future development.
 *
 * @warning This function will throw a runtime error if the database contains
 *          non-zero MPI interface data, indicating unsupported functionality.
 *
 * @code
 * // Example usage in MESHFEM3D database reading
 * std::ifstream database_stream("proc000000_mesh.bin", std::ios::binary);
 *
 * // Read other mesh components first...
 *
 * // Read MPI interfaces (currently validates only)
 * specfem::io::mesh::impl::fortran::dim3::read_mpi_interfaces(database_stream);
 * @endcode
 *
 * @see specfem::io::mesh::impl::fortran::dim3::mesh
 * @todo Implement full MPI interface processing for parallel 3D simulations
 */
void read_mpi_interfaces(std::ifstream &stream);

} // namespace specfem::io::mesh::impl::fortran::dim3
