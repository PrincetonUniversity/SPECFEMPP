#pragma once

#include "specfem/attenuation.hpp"
#include "specfem/mesh.hpp"
#include <string>

namespace specfem::io::mesh::impl::fortran::dim3 {

/** @brief Read a thin SPECFEM3D_GLOBE database. */
specfem::mesh::globe3d_mesh
read_globe_mesh(const std::string &database_file,
                const specfem::attenuation::Setup &attenuation_setup);

} // namespace specfem::io::mesh::impl::fortran::dim3
