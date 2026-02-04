#include "specfem/element.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem_setup.hpp"
#include <array>
#include <stdexcept>
#include <tuple>

const static std::array<std::array<type_real, 2>, 4> REF_FACE_NODES = {
  std::array<type_real, 2>{ 0.0, 0.0 }, std::array<type_real, 2>{ 1.0, 0.0 },
  std::array<type_real, 2>{ 1.0, 1.0 }, std::array<type_real, 2>{ 0.0, 1.0 }
};

template <typename ViewType>
std::array<int, 4> get_face_nodes(const specfem::mesh_entity::dim3::type &face,
                                  const ViewType &element) {

  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                      face)) {
    throw std::runtime_error("The provided entity is not a face.");
  }

  auto nodes = specfem::mesh_entity::nodes_on_orientation(face);

  if (nodes.size() != 4) {
    throw std::runtime_error("A face must have exactly 4 nodes.");
  }

  // Return the control node indices for the specified face
  return { element(nodes[0]), element(nodes[1]), element(nodes[2]),
           element(nodes[3]) };
}

template <typename ViewType>
std::array<int, 2> get_edge_nodes(const specfem::mesh_entity::dim3::type &edge,
                                  const ViewType &element) {

  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::edges,
                                      edge)) {
    throw std::runtime_error("The provided entity is not an edge.");
  }

  auto nodes = specfem::mesh_entity::nodes_on_orientation(edge);

  if (nodes.size() != 2) {
    throw std::runtime_error("An edge must have exactly 2 nodes.");
  }

  // Return the control node indices for the specified edge
  return { element(nodes[0]), element(nodes[1]) };
}

/**
 * @brief Computes the permutation of face nodes to match a reference face
 *
 * This function determines the permutation of the given face nodes such that
 * they match the order of the reference face nodes. The reference face is
 * assumed to be defined in a standard counter-clockwise order starting from
 * the bottom-left corner.
 *
 * @param face_nodes Array of 4 integers representing the nodes of the face
 * @param reference_face Array of 4 integers representing the reference face
 * nodes
 *
 * @return std::array<int, 4> An array representing the permutation indices
 *
 * @throws std::runtime_error if a reference face node is not found in face
 * nodes
 *
 * @note The function assumes that both face_nodes and reference_face contain
 *       exactly 4 unique node indices.
 */
std::array<int, 4>
compute_face_permutation(const std::array<int, 4> &face_nodes,
                         const std::array<int, 4> &reference_face) {
  // Find a permutation such that face_nodes[permutation[i]] ==
  // reference_face[i]
  auto it = std::find(face_nodes.begin(), face_nodes.end(), reference_face[0]);
  if (it == face_nodes.end()) {
    throw std::runtime_error("Reference face node not found in face nodes.");
  }

  // We assume that the normal from 2 faces point outward of the element they
  // are a part of i.e. in opposite directions Thus, we need to reverse the
  // order of the nodes when computing the permutation
  int start_index = std::distance(face_nodes.begin(), it);
  std::array<int, 4> permutation;
  for (int i = 0; i < 4; ++i) {
    permutation[i] = (4 + (start_index - i)) % 4;
  }

  // Verify the permutation
  for (int i = 0; i < 4; ++i) {
    if (face_nodes[permutation[i]] != reference_face[i]) {
      throw std::runtime_error("Invalid permutation computed.");
    }
  }
  return permutation;
}

/**
 * @brief Performs an affine transformation to map coordinates between permuted
 * reference faces
 *
 * This function computes an affine transformation that maps coordinates from a
 * reference face configuration to a permuted face configuration. An affine
 * transformation is a geometric transformation that preserves points, straight
 * lines, and planes, and can include translation, rotation, scaling, and
 * shearing.
 *
 * The affine transformation is defined mathematically as:
 *
 * \f[
 * \begin{pmatrix} x' \\ y' \end{pmatrix} = A \begin{pmatrix} x \\ y
 * \end{pmatrix} + b
 * \f]
 *
 * where:
 * - \f$A\f$ is a 2x2 transformation matrix
 * - \f$b\f$ is a 2D translation vector
 * - \f$(x, y)\f$ are the input coordinates (j, i)
 * - \f$(x', y')\f$ are the transformed coordinates (j', i')
 *
 * The transformation matrix A and translation vector b are computed by solving:
 * \f[
 * A = \begin{pmatrix}
 * t_1^x - t_0^x & t_3^x - t_0^x \\
 * t_1^y - t_0^y & t_3^y - t_0^y
 * \end{pmatrix}
 * \f]
 * \f[
 * b = \begin{pmatrix} t_0^x \\ t_0^y \end{pmatrix}
 * \f]
 *
 * where \f$t_i\f$ represents the i-th node of the permuted reference face.
 *
 * @param permutation Array of 4 integers defining the permutation of reference
 * face nodes
 * @param j First coordinate (typically normalized z or y coordinate depending
 * on face orientation)
 * @param i Second coordinate (typically normalized y or x coordinate depending
 * on face orientation)
 *
 * @return std::pair<type_real, type_real> The transformed coordinates (j', i')
 *
 * @note Input coordinates j and i should be normalized to [0,1] range
 * @note The function assumes a quadrilateral face with 4 nodes arranged in
 * counter-clockwise order
 */
std::pair<type_real, type_real>
affine_transform(const std::array<int, 4> &permutation, const type_real j,
                 const type_real i) {
  std::array<std::array<type_real, 2>, 2> A;
  std::array<type_real, 2> b;

  const std::array<std::array<type_real, 2>, 4> t = {
    REF_FACE_NODES[permutation[0]], REF_FACE_NODES[permutation[1]],
    REF_FACE_NODES[permutation[2]], REF_FACE_NODES[permutation[3]]
  };

  // Set up the linear system to solve for A and b
  A[0][0] = t[1][0] - t[0][0];
  A[0][1] = t[3][0] - t[0][0];
  A[1][0] = t[1][1] - t[0][1];
  A[1][1] = t[3][1] - t[0][1];

  b[0] = t[0][0];
  b[1] = t[0][1];

  // Solve for (j', i')
  type_real i_prime = A[0][0] * i + A[0][1] * j + b[0];
  type_real j_prime = A[1][0] * i + A[1][1] * j + b[1];
  return { j_prime, i_prime };
}

extern int edge_transform(const std::array<int, 2> &from_nodes,
                          const std::array<int, 2> &to_nodes, const int index,
                          const int ngll);

std::tuple<int, int, int> specfem::connections::
    connection_mapping<specfem::element::dimension_tag::dim3>::map_coordinates(
        const specfem::mesh_entity::dim3::type &from,
        const specfem::mesh_entity::dim3::type &to, const int iz, const int iy,
        const int ix) const {

  // Face to face mapping
  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces, from) &&
      specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces, to)) {
    // Implement face-to-face mapping logic here
    // This is a placeholder implementation and should be replaced with actual
    // logic
    auto face1_nodes = get_face_nodes(from, element1);
    auto face2_nodes = get_face_nodes(to, element2);

    // Compute the permutation to align face1 with face2
    auto perm = compute_face_permutation(face1_nodes, face2_nodes);

    // cordinate permutation

    // Get the (j, i) coordinates on the 'from' face
    const auto [j, i] = [=]() {
      switch (from) {
      case specfem::mesh_entity::dim3::type::left:
        return std::make_pair(static_cast<type_real>(iy) / (nglly - 1),
                              static_cast<type_real>(iz) / (ngllz - 1));
      case specfem::mesh_entity::dim3::type::right:
        return std::make_pair(static_cast<type_real>(iz) / (ngllz - 1),
                              static_cast<type_real>(iy) / (nglly - 1));
      case specfem::mesh_entity::dim3::type::front:
        return std::make_pair(static_cast<type_real>(iz) / (ngllz - 1),
                              static_cast<type_real>(ix) / (ngllx - 1));
      case specfem::mesh_entity::dim3::type::back:
        return std::make_pair(static_cast<type_real>(ix) / (ngllx - 1),
                              static_cast<type_real>(iz) / (ngllz - 1));
      case specfem::mesh_entity::dim3::type::bottom:
        return std::make_pair(static_cast<type_real>(ix) / (ngllx - 1),
                              static_cast<type_real>(iy) / (nglly - 1));
      case specfem::mesh_entity::dim3::type::top:
        return std::make_pair(static_cast<type_real>(iy) / (nglly - 1),
                              static_cast<type_real>(ix) / (ngllx - 1));
      default:
        throw std::runtime_error("Invalid face orientation.");
      }
    }();

    const auto [j_prime, i_prime] = affine_transform(perm, j, i);
    // Map (j', i') to (iz, iy, ix) on the 'to' face
    return [=](const type_real j_prime, const type_real i_prime) {
      switch (to) {
      case specfem::mesh_entity::dim3::type::left:
        return std::make_tuple(static_cast<int>(i_prime * (ngllz - 1)),
                               static_cast<int>(j_prime * (nglly - 1)), 0);
      case specfem::mesh_entity::dim3::type::right:
        // X-axis faces
        return std::make_tuple(static_cast<int>(j_prime * (ngllz - 1)),
                               static_cast<int>(i_prime * (nglly - 1)),
                               ngllx - 1);
      case specfem::mesh_entity::dim3::type::front:
        return std::make_tuple(static_cast<int>(j_prime * (ngllz - 1)), 0,
                               static_cast<int>(i_prime * (ngllx - 1)));
      case specfem::mesh_entity::dim3::type::back:
        // Y-axis faces
        return std::make_tuple(static_cast<int>(i_prime * (ngllz - 1)),
                               nglly - 1,
                               static_cast<int>(j_prime * (ngllx - 1)));
      case specfem::mesh_entity::dim3::type::top:
        // Z-axis faces
        return std::make_tuple(ngllz - 1,
                               static_cast<int>(j_prime * (nglly - 1)),
                               static_cast<int>(i_prime * (ngllx - 1)));
      case specfem::mesh_entity::dim3::type::bottom:
        // Z-axis faces
        return std::make_tuple(0, static_cast<int>(i_prime * (nglly - 1)),
                               static_cast<int>(j_prime * (ngllx - 1)));
      default:
        throw std::runtime_error("Invalid face orientation.");
      }
    }(j_prime, i_prime);
  }

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::edges, from) &&
      specfem::mesh_entity::contains(specfem::mesh_entity::dim3::edges, to)) {

    auto edge1_nodes = get_edge_nodes(from, element1);
    auto edge2_nodes = get_edge_nodes(to, element2);

    const auto [i, n] = [=]() {
      switch (from) {
      case specfem::mesh_entity::dim3::type::front_bottom:
      case specfem::mesh_entity::dim3::type::front_top:
      case specfem::mesh_entity::dim3::type::back_bottom:
      case specfem::mesh_entity::dim3::type::back_top:
        return std::make_pair(ix, ngllx);
      case specfem::mesh_entity::dim3::type::bottom_left:
      case specfem::mesh_entity::dim3::type::top_left:
      case specfem::mesh_entity::dim3::type::bottom_right:
      case specfem::mesh_entity::dim3::type::top_right:
        return std::make_pair(iy, nglly);
      case specfem::mesh_entity::dim3::type::front_left:
      case specfem::mesh_entity::dim3::type::front_right:
      case specfem::mesh_entity::dim3::type::back_left:
      case specfem::mesh_entity::dim3::type::back_right:
        return std::make_pair(iz, ngllz);
      default:
        throw std::runtime_error("Invalid edge orientation.");
      }
    }();

    const int i_prime = edge_transform(edge1_nodes, edge2_nodes, i, n);

    return [=](const int i_prime) {
      switch (to) {
      case specfem::mesh_entity::dim3::type::front_bottom:
        return std::make_tuple(0, 0, i_prime);
      case specfem::mesh_entity::dim3::type::front_top:
        return std::make_tuple(ngllz - 1, 0, i_prime);
      case specfem::mesh_entity::dim3::type::back_bottom:
        return std::make_tuple(0, nglly - 1, i_prime);
      case specfem::mesh_entity::dim3::type::back_top:
        return std::make_tuple(ngllz - 1, nglly - 1, i_prime);
      case specfem::mesh_entity::dim3::type::bottom_left:
        return std::make_tuple(0, i_prime, 0);
      case specfem::mesh_entity::dim3::type::top_left:
        return std::make_tuple(ngllz - 1, i_prime, 0);
      case specfem::mesh_entity::dim3::type::bottom_right:
        return std::make_tuple(0, i_prime, ngllx - 1);
      case specfem::mesh_entity::dim3::type::top_right:
        return std::make_tuple(ngllz - 1, i_prime, ngllx - 1);
      case specfem::mesh_entity::dim3::type::front_left:
        return std::make_tuple(i_prime, 0, 0);
      case specfem::mesh_entity::dim3::type::front_right:
        return std::make_tuple(i_prime, 0, ngllx - 1);
      case specfem::mesh_entity::dim3::type::back_left:
        return std::make_tuple(i_prime, nglly - 1, 0);
      case specfem::mesh_entity::dim3::type::back_right:
        return std::make_tuple(i_prime, nglly - 1, ngllx - 1);
      default:
        throw std::runtime_error("Invalid edge orientation.");
      }
    }(i_prime);
  }
  throw std::runtime_error(
      "Mapping between the specified mesh entities is not implemented.");
  return std::make_tuple(0, 0, 0);
}

std::tuple<int, int, int> specfem::connections::
    connection_mapping<specfem::element::dimension_tag::dim3>::map_coordinates(
        const specfem::mesh_entity::dim3::type &from,
        const specfem::mesh_entity::dim3::type &to) const {

  if (!(specfem::mesh_entity::contains(specfem::mesh_entity::dim3::corners,
                                       from) &&
        specfem::mesh_entity::contains(specfem::mesh_entity::dim3::corners,
                                       to)))
    throw std::runtime_error("Both entities must be corners for this mapping.");

  return specfem::mesh_entity::element(ngllz, nglly, ngllx).map_coordinates(to);
}
