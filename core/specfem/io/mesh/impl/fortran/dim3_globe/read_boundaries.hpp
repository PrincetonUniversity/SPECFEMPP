#pragma once

#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3_globe {

/** @brief Read one globe boundary surface section. */
specfem::mesh::globe_boundary_surface read_surface(std::ifstream &stream,
                                                   const int nspec);

/** @brief Read globe boundary surfaces and populate raw mesh boundaries. */
void read_boundaries(std::ifstream &stream, specfem::mesh::globe3d_mesh &mesh);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
