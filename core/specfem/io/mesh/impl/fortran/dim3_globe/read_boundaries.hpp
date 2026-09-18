#pragma once

#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3_globe {

/**
 * @brief Read one named boundary surface from a thin globe database.
 *
 * A surface section contains a face count followed by one-based element ids and
 * SPECFEM3D_GLOBE face ids. Element ids are converted to zero-based SPECFEM++
 * indices; face ids are stored as @c specfem::mesh_entity::dim3::type values.
 *
 * @param stream Input stream positioned at a boundary surface section
 * @param nspec Number of local elements used to validate element ids
 * @return Boundary element/face pairs for the surface
 * @throws std::runtime_error if the face count, element ids, or face ids are
 *         invalid
 */
specfem::mesh::globe_boundary_surface read_surface(std::ifstream &stream,
                                                   const int nspec);

/**
 * @brief Read globe boundary surfaces and populate generic mesh boundaries.
 *
 * The thin database stores free surface, CMB, ICB, and ocean-load surfaces in
 * that order. All surfaces are retained in @c mesh.globe; the free surface is
 * also copied into @c mesh.boundaries as the acoustic free-surface boundary
 * used by the generic 3-D assembly path. Globe databases currently do not
 * provide absorbing boundaries, so that part of @c mesh.boundaries is
 * initialized empty.
 *
 * @param stream Input stream positioned at the first globe boundary surface
 * @param mesh Globe mesh whose globe surfaces and generic boundaries are set
 * @throws std::runtime_error if any surface section is malformed
 */
void read_boundaries(std::ifstream &stream, specfem::mesh::globe3d_mesh &mesh);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
