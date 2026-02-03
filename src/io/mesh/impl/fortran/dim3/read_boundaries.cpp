#include "io/mesh/impl/fortran/dim3/read_boundaries.hpp"
#include "io/fortranio/interface.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

/**
 * @brief Calculates the characteristic length of an element in 3D space.
 *
 * This function computes the characteristic length of a 3D element by finding
 * the bounding box of all its control nodes and returning the maximum dimension
 * (length, width, or height) of that bounding box.
 *
 * @param control_nodes Reference to the control nodes structure containing
 *                      node coordinates and indexing information for 3D mesh
 * elements
 * @param element_index Index of the element for which to calculate the
 * characteristic length
 *
 * @return type_real The characteristic length of the element, defined as the
 * maximum of the three bounding box dimensions (max_x-min_x, max_y-min_y,
 * max_z-min_z)
 *
 * @note The characteristic length is commonly used in finite element methods
 *       for mesh quality assessment and numerical stability calculations.
 */
type_real characteristic_length(
    const specfem::mesh::control_nodes<specfem::dimension::type::dim3>
        &control_nodes,
    const int element_index) {
  const int nodes_per_element = control_nodes.ngnod;
  const auto &coordinates = control_nodes.coordinates;
  const auto &control_node_index = control_nodes.control_node_index;
  type_real min_x = std::numeric_limits<type_real>::max();
  type_real max_x = std::numeric_limits<type_real>::lowest();
  type_real min_y = min_x, max_y = max_x;
  type_real min_z = min_x, max_z = max_x;

  for (int inode = 0; inode < nodes_per_element; ++inode) {
    const int node_id = control_node_index(element_index, inode);
    const type_real x = coordinates(node_id, 0);
    const type_real y = coordinates(node_id, 1);
    const type_real z = coordinates(node_id, 2);

    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
  }

  return std::max({ max_x - min_x, max_y - min_y, max_z - min_z });
}

/**
 * @brief Identifies which face of a 3D element corresponds to a given set of
 * nodes
 *
 * This function determines which face (bottom, top, front, back, left, or
 * right) of a hexahedral element matches the provided face nodes by comparing
 * midpoint coordinates. The function validates that the identified face is
 * geometrically consistent with the input nodes within a reasonable tolerance.
 *
 * @param control_nodes The control nodes structure containing element
 * connectivity and coordinate information for the mesh
 * @param element_index The index of the element being analyzed
 * @param face_nodes Vector of node indices that define the face to identify
 *
 * @return The face type (bottom, top, front, back, left, or right) that
 *         corresponds to the given nodes
 *
 * @throws std::runtime_error If no face can be matched within acceptable
 * tolerance or if an invalid face type is encountered
 */
specfem::mesh_entity::dim3::type find_face_from_nodes(
    const specfem::mesh::control_nodes<specfem::dimension::type::dim3>
        &control_nodes,
    const int element_index, const std::vector<int> &face_nodes) {

  const int nodes_per_element = control_nodes.ngnod;
  const auto &coordinates = control_nodes.coordinates;
  const auto &control_node_index = control_nodes.control_node_index;

  // find coordinates of midpoint for each face
  std::unordered_map<
      specfem::mesh_entity::dim3::type,
      specfem::point::global_coordinates<specfem::dimension::type::dim3> >
      face_midpoints;

  auto corner_id_on_face =
      [&](specfem::mesh_entity::dim3::type face) -> std::vector<int> {
    switch (face) {
    case specfem::mesh_entity::dim3::type::bottom:
      return { 0, 1, 2, 3 };
    case specfem::mesh_entity::dim3::type::top:
      return { 4, 5, 6, 7 };
    case specfem::mesh_entity::dim3::type::front:
      return { 0, 1, 5, 4 };
    case specfem::mesh_entity::dim3::type::back:
      return { 3, 2, 6, 7 };
    case specfem::mesh_entity::dim3::type::left:
      return { 0, 3, 7, 4 };
    case specfem::mesh_entity::dim3::type::right:
      return { 1, 2, 6, 5 };
    default:
      throw std::runtime_error("Invalid face type");
    }
  };

  for (const auto &face_type : specfem::mesh_entity::dim3::faces) {
    const auto corner_ids = corner_id_on_face(face_type);
    specfem::point::global_coordinates<specfem::dimension::type::dim3> midpoint{
      0.0, 0.0, 0.0
    };
    for (const auto corner_id : corner_ids) {
      const int node_id = control_node_index(element_index, corner_id);
      midpoint.x += coordinates(node_id, 0);
      midpoint.y += coordinates(node_id, 1);
      midpoint.z += coordinates(node_id, 2);
    }
    midpoint.x /= corner_ids.size();
    midpoint.y /= corner_ids.size();
    midpoint.z /= corner_ids.size();
    face_midpoints[face_type] = midpoint;
  }

  // find midpoint of face_nodes
  specfem::point::global_coordinates<specfem::dimension::type::dim3>
      face_nodes_midpoint{ 0.0, 0.0, 0.0 };
  for (const auto node : face_nodes) {
    face_nodes_midpoint.x += coordinates(node, 0);
    face_nodes_midpoint.y += coordinates(node, 1);
    face_nodes_midpoint.z += coordinates(node, 2);
  }
  face_nodes_midpoint.x /= face_nodes.size();
  face_nodes_midpoint.y /= face_nodes.size();
  face_nodes_midpoint.z /= face_nodes.size();

  // find face whose midpoint is closest to face_nodes_midpoint
  const auto closest_face_iter = std::min_element(
      face_midpoints.begin(), face_midpoints.end(),
      [&face_nodes_midpoint](const auto &a, const auto &b) {
        const auto distance_a =
            specfem::point::distance(a.second, face_nodes_midpoint);
        const auto distance_b =
            specfem::point::distance(b.second, face_nodes_midpoint);
        return distance_a < distance_b;
      });
  const auto closest_face = closest_face_iter->first;

  // Find characteristic length of the element
  const auto lc = characteristic_length(control_nodes, element_index);

  // Check if the closest face is indeed close enough
  const auto min_distance =
      specfem::point::distance(closest_face_iter->second, face_nodes_midpoint);
  if (min_distance > 1e-3 * lc) {
    throw std::runtime_error(
        "Could not find matching face for absorbing boundary. Element " +
        std::to_string(element_index) + ": closest face distance " +
        std::to_string(min_distance) + " exceeds tolerance " +
        std::to_string(1e-3 * lc) +
        " (characteristic length: " + std::to_string(lc) + ")");
  }

  return closest_face;
}

specfem::mesh::boundaries<specfem::dimension::type::dim3>
specfem::io::mesh::impl::fortran::dim3::read_boundaries(
    std::ifstream &stream, const int nspec,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim3>
        &control_nodes) {

  int boundary_number;
  std::array<int, 6> nfaces_per_direction;

  using BoundaryType =
      specfem::mesh::boundaries<specfem::dimension::type::dim3>;

  using face_direction = BoundaryType::FaceDirection;

  specfem::io::fortran_read_line(
      stream, &boundary_number,
      &nfaces_per_direction[static_cast<int>(face_direction::X_MIN) - 1]);
  if (boundary_number != static_cast<int>(face_direction::X_MIN)) {
    throw std::runtime_error(
        "Invalid database format: expected boundary number 1");
  }

  specfem::io::fortran_read_line(
      stream, &boundary_number,
      &nfaces_per_direction[static_cast<int>(face_direction::X_MAX) - 1]);
  if (boundary_number != static_cast<int>(face_direction::X_MAX)) {
    throw std::runtime_error(
        "Invalid database format: expected boundary number 2");
  }

  specfem::io::fortran_read_line(
      stream, &boundary_number,
      &nfaces_per_direction[static_cast<int>(face_direction::Y_MIN) - 1]);
  if (boundary_number != static_cast<int>(face_direction::Y_MIN)) {
    throw std::runtime_error(
        "Invalid database format: expected boundary number 3");
  }

  specfem::io::fortran_read_line(
      stream, &boundary_number,
      &nfaces_per_direction[static_cast<int>(face_direction::Y_MAX) - 1]);
  if (boundary_number != static_cast<int>(face_direction::Y_MAX)) {
    throw std::runtime_error(
        "Invalid database format: expected boundary number 4");
  }

  specfem::io::fortran_read_line(
      stream, &boundary_number,
      &nfaces_per_direction[static_cast<int>(face_direction::Z_MIN) - 1]);
  if (boundary_number != static_cast<int>(face_direction::Z_MIN)) {
    throw std::runtime_error(
        "Invalid database format: expected boundary number 5");
  }

  specfem::io::fortran_read_line(
      stream, &boundary_number,
      &nfaces_per_direction[static_cast<int>(face_direction::Z_MAX) - 1]);
  if (boundary_number != static_cast<int>(face_direction::Z_MAX)) {
    throw std::runtime_error(
        "Invalid database format: expected boundary number 6");
  }

  const int total_nfaces = std::accumulate(nfaces_per_direction.begin(),
                                           nfaces_per_direction.end(), 0);

  BoundaryType boundaries(total_nfaces, nspec);

  const int nnodes_on_face = (control_nodes.ngnod == 8) ? 4 : 9;

  int index = 0;
  for (int dir = 0; dir < 6; ++dir) {
    const int num_faces = nfaces_per_direction[dir];
    const face_direction face_dir = static_cast<face_direction>(dir + 1);
    if (num_faces > 0) {
      for (int iface = 0; iface < num_faces; ++iface) {
        int element_index;
        std::vector<int> face_nodes(nnodes_on_face);
        specfem::io::fortran_read_line(stream, &element_index, &face_nodes);
        // Decrement to convert to zero-based indexing
        for (auto &node : face_nodes) {
          node -= 1;
        }
        const auto face =
            find_face_from_nodes(control_nodes, element_index - 1, face_nodes);
        boundaries.index_mapping(index) = element_index - 1;
        boundaries.face_type(index) = face;
        boundaries.face_direction(index) = face_dir;
        ++index;
      }
    }
  }

  if (index != total_nfaces) {
    throw std::runtime_error("Error reading absorbing boundaries. Expected " +
                             std::to_string(total_nfaces) +
                             " faces, but read " + std::to_string(index) +
                             " faces.");
  }

  return boundaries;
}
