#pragma once

#include "mesh/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d {
/**
 * @brief Read boundary face information from MESHFEM3D database files
 *
 * Reads binary Fortran-formatted boundary data and constructs a
 * boundaries object containing face indices and face types(bottom, top, front,
 * back, left, right).
 *
 * @param stream Input file stream positioned at boundary data
 * section, opened in binary mode
 * @param nspec Total number of spectral elements in the mesh
 * @param control_nodes ControlNodes object containing node coordinates for face
 * matching
 * @param mpi MPI communication interface for parallel processing
 *
 * @return A Boundaries object with face count, element indices, and face
 * types
 *
 * @throws std::runtime_error If database format is invalid or face matching
 * fails
 */
specfem::mesh::meshfem3d::Boundaries<specfem::dimension::type::dim3>
read_boundaries(
    std::ifstream &stream, const int nspec,
    const specfem::mesh::meshfem3d::ControlNodes<specfem::dimension::type::dim3>
        &control_nodes);

} // namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d
