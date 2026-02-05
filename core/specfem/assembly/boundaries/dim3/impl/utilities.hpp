#pragma once

#include "enumerations/interface_tags.hpp"
#include "specfem/datatype.hpp"
#include "specfem/point.hpp"
#include <array>
#include <tuple>

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Check if a point is on the boundary for a given boundary type
 *
 * @param type The boundary type to check
 * @param iz Z-direction index
 * @param iy Y-direction index
 * @param ix X-direction index
 * @param ngllz Number of GLL points in Z direction
 * @param nglly Number of GLL points in Y direction
 * @param ngllx Number of GLL points in X direction
 * @return true if point is on the specified boundary, false otherwise
 */
bool is_on_boundary(specfem::mesh_entity::dim3::type type, int iz, int iy,
                    int ix, int ngllz, int nglly, int ngllx);

/**
 * @brief Get the boundary face normal and weight for boundary integration
 *
 * @param type The boundary type
 * @param weights Array of quadrature weights
 * @param point_jacobian_matrix Jacobian matrix at the point
 * @return Tuple containing face normal vector and integration weight
 */
std::tuple<std::array<type_real, 3>, type_real> get_boundary_face_and_weight(
    specfem::mesh_entity::dim3::type type,
    const std::array<type_real, 3> &weights,
    const specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                          true, false> &point_jacobian_matrix);

} // namespace specfem::assembly::boundaries_impl
