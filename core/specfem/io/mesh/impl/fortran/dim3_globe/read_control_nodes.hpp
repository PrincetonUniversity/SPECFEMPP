#pragma once

#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3_globe {

/**
 * @brief Read globe control-node and reference-coordinate arrays.
 *
 * The stream must be positioned at the global anchor-node count. Physical
 * coordinates are copied into @c mesh.control_nodes and coordinate bounds are
 * updated. If @c mesh.globe.has_reference_geometry is true, the following
 * record is read as reference coordinates; otherwise the physical coordinates
 * are reused as the reference coordinates required by globe property setup.
 *
 * @param stream Input stream positioned at the control-node coordinate section
 * @param mesh Globe mesh whose control-node coordinates are populated
 * @param ngnod Number of anchor nodes per element
 * @return Number of unique global anchor nodes in the database
 * @throws std::runtime_error if the database reports a non-positive node count
 */
int read_control_node_coordinates(std::ifstream &stream,
                                  specfem::mesh::globe3d_mesh &mesh,
                                  const int ngnod);

/**
 * @brief Read per-element control-node connectivity from a globe database.
 *
 * The thin database stores one-based global anchor-node ids in the same local
 * hex27 order used by SPECFEM++. This function converts ids to zero-based
 * indices and fills @c mesh.control_nodes.control_node_index.
 *
 * @param stream Input stream positioned at the element connectivity section
 * @param mesh Globe mesh whose control-node index mapping is populated
 * @param ngnod Number of anchor nodes per element
 * @param nnode Number of global anchor nodes used to validate connectivity
 * @throws std::runtime_error if any anchor-node id is out of range
 */
void read_control_node_indices(std::ifstream &stream,
                               specfem::mesh::globe3d_mesh &mesh,
                               const int ngnod, const int nnode);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
