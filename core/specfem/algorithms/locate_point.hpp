#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/point.hpp"
#include "specfem/point/global_coordinates.hpp"
#include "specfem/point/local_coordinates.hpp"
#include "specfem/setup.hpp"
#include <type_traits>

namespace specfem {
namespace algorithms {

/**
 * @brief Convert global coordinates to local coordinates in a 2D mesh
 *
 * @param coordinates Global coordinates to convert
 * @param mesh 2D spectral element mesh
 * @return Local coordinates within the containing element
 */
specfem::point::local_coordinates<specfem::element::dimension_tag::dim2>
locate_point(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh);

/**
 * @brief Convert local coordinates to global coordinates in a 2D mesh
 *
 * @param coordinates Local coordinates to convert
 * @param mesh 2D spectral element mesh
 * @return Global coordinates in physical space
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
locate_point(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh);

/**
 * @brief Convert local coordinates to global coordinates in a 2D mesh using
 * team parallelism
 *
 * @param team_member Kokkos team member for parallel execution
 * @param coordinates Local coordinates to convert
 * @param mesh 2D spectral element mesh
 * @return Global coordinates in physical space
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
locate_point(
    const Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type
        &team_member,
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh);

/**
 * @brief Convert global coordinates to local coordinates in a 3D mesh
 *
 * @param coordinates Global coordinates to convert
 * @param mesh 3D spectral element mesh
 * @return Local coordinates within the containing element
 */
specfem::point::local_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

/**
 * @brief Convert local coordinates to global coordinates in a 3D mesh
 *
 * @param coordinates Local coordinates to convert
 * @param mesh 3D spectral element mesh
 * @return Global coordinates in physical space
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

/**
 * @brief Convert local coordinates to global coordinates in a 3D mesh using
 * team parallelism
 *
 * @param team_member Kokkos team member for parallel execution
 * @param coordinates Local coordinates to convert
 * @param mesh 3D spectral element mesh
 * @return Global coordinates in physical space
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type
        &team_member,
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

/**
 * @brief Given an edge (ispec, constraint), finds the best fit local coordinate
 * on that edge to the given global coordinates. Coordinates will be clamped to
 * [-1,1], even if a point outside that range is a better fit. In such a case,
 * the second return value will be false.
 *
 * @param coordinates - global coordinates to match to
 * @param mesh - assembly::mesh struct
 * @param ispec - element index whose local coordinates to find
 * @param constraint - edge to compute for
 * @return std::pair<type_real,bool> - the edge local coordinate and whether or
 * not the minimum found is a critical point (false is returned if the best fit
 * coordinate is out of bounds).
 */
std::pair<type_real, bool> locate_point_on_edge(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::dim2::type &constraint);
/**
 * @brief Convert edge coordinate to global coordinates
 *
 * Given an edge (ispec, constraint) and the coordinate along it, finds
 * the global coordinates.
 *
 * @param coordinate Local coordinate along edge
 * @param mesh 2D spectral element mesh
 * @param ispec Element index whose local coordinates to find
 * @param constraint Edge to compute for
 * @return Global coordinates of the point
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
locate_point_on_edge(
    const type_real &coordinate,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::dim2::type &constraint);

/**
 * @brief A type for the local coordinates along a given face (codimension 1)
 *
 * @tparam dimension_tag the dimension of the element.
 */
template <specfem::element::dimension_tag dimension_tag>
using facial_coordinate_type =
    std::conditional_t<dimension_tag == specfem::element::dimension_tag::dim2,
                       type_real, std::pair<type_real, type_real> >;

/**
 * @brief Given a face (ispec, constraint), finds the best fit local coordinate
 * on that face to the given global coordinates. Coordinates will be clamped to
 * [-1,1], even if a point outside that range is a better fit. In such a case,
 * the second return value will be false.
 *
 * @param coordinates - global coordinates to match to
 * @param mesh - assembly::mesh struct
 * @param ispec - element index whose local coordinates to find
 * @param constraint - edge to compute for
 * @return std::pair<facial_coordinate_type,bool> - the face local coordinate
 * and whether or not the minimum found is a critical point (false is returned
 * if the best fit coordinate is out of bounds).
 */
template <specfem::element::dimension_tag dimension_tag>
std::pair<facial_coordinate_type<dimension_tag>, bool> locate_point_on_face(
    const specfem::point::global_coordinates<dimension_tag> &coordinates,
    const specfem::assembly::mesh<dimension_tag> &mesh, const int &ispec,
    const specfem::mesh_entity::type<dimension_tag> &constraint);

/**
 * @brief Convert face coordinate to global coordinates
 *
 * Given a face (ispec, constraint) and the coordinates along it, finds
 * the global coordinates.
 *
 * @param coordinates Local coordinate along edge
 * @param mesh 2D spectral element mesh
 * @param ispec Element index whose local coordinates to find
 * @param constraint Edge to compute for
 * @return Global coordinates of the point
 */
template <specfem::element::dimension_tag dimension_tag>
specfem::point::global_coordinates<dimension_tag> locate_point_on_face(
    const facial_coordinate_type<dimension_tag> &coordinates,
    const specfem::assembly::mesh<dimension_tag> &mesh, const int &ispec,
    const specfem::mesh_entity::type<dimension_tag> &constraint);

} // namespace algorithms
} // namespace specfem

/**
 * @defgroup AlgorithmsLocatePoint Point Location Algorithms
 * @ingroup Algorithms
 */
