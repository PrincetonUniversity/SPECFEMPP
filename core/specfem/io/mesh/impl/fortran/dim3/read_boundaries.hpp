#pragma once

#include "specfem/mesh.hpp"

#include <fstream>
#include <vector>

namespace specfem::io::mesh::impl::fortran::dim3_impl {

/// Matches BOUNDARY_HEADER_MAGIC in the Fortran meshers. Negative so it cannot
/// be mistaken for the first record of older layouts.
constexpr int boundary_header_magic = -2154;
constexpr int boundary_header_version = 1;

/**
 * @brief Compute the characteristic element length for tolerance checks.
 *
 * @param control_nodes Control-node data for the 3D mesh
 * @param element_index Zero-based element index
 * @return Maximum bounding-box dimension of the element
 */
type_real characteristic_length(
    const specfem::mesh::control_nodes<specfem::element::dimension_tag::dim3>
        &control_nodes,
    const int element_index);

/**
 * @brief Identify which face of an element matches a set of nodes.
 *
 * @param control_nodes Control-node data for the 3D mesh
 * @param element_index Zero-based element index
 * @param face_nodes    Zero-based node indices on the face
 * @return Face type (bottom, top, front, back, left, right)
 * @throws std::runtime_error if no face can be matched within tolerance
 */
specfem::mesh_entity::dim3::type find_face_from_nodes(
    const specfem::mesh::control_nodes<specfem::element::dimension_tag::dim3>
        &control_nodes,
    const int element_index, const std::vector<int> &face_nodes);

} // namespace specfem::io::mesh::impl::fortran::dim3_impl

namespace specfem::io::mesh::impl::fortran::dim3 {

/**
 * @brief Read boundary face information from the SPECFEM++ 3D mesh database.
 *
 * Reads the boundary header and six face-direction sections from the binary
 * database stream, then classifies faces into:
 * - **absorbing_boundary**: sections 1-4 (X_MIN, X_MAX, Y_MIN, Y_MAX) are
 *   always absorbing; section 5 (Z_MIN) and section 6 (Z_MAX) are absorbing
 *   when BOTTOM_FREE_SURFACE and TOP_FREE_SURFACE are false, respectively
 * - **acoustic_free_surface**: section 5 when BOTTOM_FREE_SURFACE=true;
 *   section 6 when TOP_FREE_SURFACE=true
 * - **injection_surface**: faces on the sides the mesher marked for injection
 *
 * Older databases without the versioned header have no injection faces.
 *
 * @param stream        Input stream positioned at the boundary data section
 * @param nspec         Total number of spectral elements in the mesh
 * @param control_nodes Control-node data used for face identification
 * @return              A @c boundaries<dim3> object with the sub-structs
 *                      populated
 * @throws std::runtime_error on invalid database format, an unsupported
 * header version or injection kind, an injection side that is a free surface,
 * or face-matching failure
 */
specfem::mesh::boundaries<specfem::element::dimension_tag::dim3>
read_boundaries(
    std::ifstream &stream, const int nspec,
    const specfem::mesh::control_nodes<specfem::element::dimension_tag::dim3>
        &control_nodes);

} // namespace specfem::io::mesh::impl::fortran::dim3
