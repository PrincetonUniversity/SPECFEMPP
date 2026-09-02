#pragma once

#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3_globe {

/**
 * @brief Read globe control-node and reference-coordinate arrays.
 *
 * @param stream Input stream positioned at the control-node coordinate section
 * @param mesh Globe mesh being populated
 * @param ngnod Number of anchor nodes per element
 * @return Number of unique control nodes in the database
 */
int read_control_node_coordinates(std::ifstream &stream,
                                  specfem::mesh::globe3d_mesh &mesh,
                                  const int ngnod);

/** @brief Read per-element control-node indices from a globe database. */
void read_control_node_indices(std::ifstream &stream,
                               specfem::mesh::globe3d_mesh &mesh,
                               const int ngnod, const int nnode);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
