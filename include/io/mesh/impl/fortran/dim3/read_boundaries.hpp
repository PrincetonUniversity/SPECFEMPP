#pragma once

#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3 {
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
 * @param control_nodes control_nodes object containing node coordinates for
 * face matching
 *
 * @return A boundaries object with face count, element indices, and face
 * types
 *
 * @throws std::runtime_error If database format is invalid or face matching
 * fails
 */
specfem::mesh::boundaries<specfem::dimension::type::dim3> read_boundaries(
    std::ifstream &stream, const int nspec,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim3>
        &control_nodes);

} // namespace specfem::io::mesh::impl::fortran::dim3
