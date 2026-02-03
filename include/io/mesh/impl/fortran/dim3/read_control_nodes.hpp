#pragma once

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3 {

/**
 * @brief Reads control nodes from a meshfem3D binary database file.
 *
 * Reads node coordinates from a SPECFEM3D meshfem3D binary file using fortran
 * unformatted record structure. File contains node count followed by node data
 * (index, x, y, z coordinates).
 *
 * @param stream Input file stream positioned at control nodes section
 * @return ControlNodes object containing node count and 3D coordinates
 * @throws std::runtime_error if file reading fails
 */
specfem::mesh::control_nodes<specfem::dimension::type::dim3>
read_control_nodes(std::ifstream &stream);

} // namespace specfem::io::mesh::impl::fortran::dim3
