#ifndef _COMPUTE_BOUNDARIES_HPP
#define _COMPUTE_BOUNDARIES_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

namespace specfem {
namespace compute {

/**
 * @namespace Defines special functions used to access structs defined in
 * compute module.
 *
 */
namespace access {

/**
 * @brief Struct to save boundary types for each element
 *
 * We store the boundary tag for every edge/node on the element
 *
 */
struct boundary_types {

  specfem::enums::element::boundary_tag top =
      specfem::enums::element::boundary_tag::none; ///< top boundary tag
  specfem::enums::element::boundary_tag bottom =
      specfem::enums::element::boundary_tag::none; ///< bottom boundary tag
  specfem::enums::element::boundary_tag left =
      specfem::enums::element::boundary_tag::none; ///< left boundary tag
  specfem::enums::element::boundary_tag right =
      specfem::enums::element::boundary_tag::none; ///< right boundary tag
  specfem::enums::element::boundary_tag bottom_right =
      specfem::enums::element::boundary_tag::none; ///< bottom right boundary
                                                   ///< tag
  specfem::enums::element::boundary_tag bottom_left =
      specfem::enums::element::boundary_tag::none; ///< bottom left boundary tag
  specfem::enums::element::boundary_tag top_right =
      specfem::enums::element::boundary_tag::none; ///< top right boundary tag
  specfem::enums::element::boundary_tag top_left =
      specfem::enums::element::boundary_tag::none; ///< top left boundary tag

  /**
   * @brief Construct a new boundary types object
   *
   */
  KOKKOS_FUNCTION boundary_types() = default;

  /**
   * @brief Update the tag for a given boundary type
   *
   * @param type Type of the boundary to update - defines an edge or node
   * @param tag Tag to update the boundary with
   */
  void update_boundary_type(const specfem::enums::boundaries::type &type,
                            const specfem::enums::element::boundary_tag &tag);
};

/**
 * @brief Evaluate if a GLL point is on a boundary of type tag
 *
 * @param tag Boundary tag to check
 * @param type Boundary type to check
 * @param iz z-index of GLL point
 * @param ix x-index of GLL point
 * @param ngllz Number of GLL points in z-direction
 * @param ngllx Number of GLL points in x-direction
 * @return bool True if the GLL point is on the boundary, false otherwise
 */
KOKKOS_FUNCTION bool
is_on_boundary(const specfem::enums::element::boundary_tag &tag,
               const specfem::compute::access::boundary_types &type,
               const int &iz, const int &ix, const int &ngllz,
               const int &ngllx);
} // namespace access

/**
 * @brief Struct to store the acoustic free surface boundary
 *
 */
struct acoustic_free_surface {

  /**
   * @brief Construct a new acoustic free surface object
   *
   * @param kmato Element to material mapping
   * @param materials Vector of materials
   * @param absorbing_boundaries Absorbing boundary object defined in mesh
   * module
   * @param acoustic_free_surface Acoustic free surface boundary object defined
   * in mesh module
   */
  acoustic_free_surface(
      const specfem::kokkos::HostView1d<int> kmato,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  int nelements; ///< Number of elements with acoustic free surface boundary
  specfem::kokkos::DeviceView1d<int> ispec;   ///< Element indices with acoustic
                                              ///< free surface boundary
  specfem::kokkos::HostMirror1d<int> h_ispec; ///< Host mirror of ispec
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types>
      type; ///< Boundary information for each element
  specfem::kokkos::HostMirror1d<specfem::compute::access::boundary_types>
      h_type; ///< Host mirror of type
};

/**
 * @brief
 *
 */
template <specfem::enums::element::type type,
          specfem::enums::element::property_tag property>
struct stacey_medium {

  /**
   * @brief Construct a new stacey medium object
   *
   */
  stacey_medium() = default;

  /**
   * @brief Construct a new stacey medium object
   *
   * @param medium Type of medium to construct
   * @param kmato Element to material mapping
   * @param materials Vector of materials
   * @param absorbing_boundaries Absorbing boundary object defined in mesh
   * module
   * @param acoustic_free_surface Acoustic free surface boundary object defined
   * in mesh module
   */
  stacey_medium(
      const specfem::kokkos::HostView1d<int> kmato,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  int nelements; ///< Number of elements with Stacey boundary
  specfem::kokkos::DeviceView1d<int> ispec;   ///< Element indices with Stacey
                                              ///< boundary
  specfem::kokkos::HostMirror1d<int> h_ispec; ///< Host mirror of ispec
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types>
      type; ///< Boundary information for each element
  specfem::kokkos::HostMirror1d<specfem::compute::access::boundary_types>
      h_type; ///< Host mirror of type
};

/**
 * @brief Struct to store stacey boundaries on the simulation domain
 *
 */
struct stacey {
  /**
   * @brief Construct a new stacey object
   *
   * @param kmato Element to material mapping
   * @param materials Vector of materials
   * @param absorbing_boundaries Absorbing boundary object defined in mesh
   * module
   * @param acoustic_free_surface Acoustic free surface boundary object defined
   * in mesh module
   */
  stacey(
      const specfem::compute::mesh::control_nodes &control_nodes,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  int nelements; ///< Number of elements with Stacey boundary
  specfem::compute::stacey_medium<
      specfem::enums::element::type::elastic,
      specfem::enums::element::property_tag::isotropic>
      elastic; ///< Elastic Stacey boundary
  specfem::compute::stacey_medium<
      specfem::enums::element::type::acoustic,
      specfem::enums::element::property_tag::isotropic>
      acoustic; ///< Acoustic Stacey boundary
};

/**
 * @brief Struct to store Stacey and Dirichlet composite boundaries
 *
 */
struct composite_stacey_dirichlet {
  /**
   * @brief Construct a new composite stacey dirichlet object
   *
   * @param kmato Element to material mapping
   * @param materials Vector of materials
   * @param absorbing_boundaries Absorbing boundary object defined in mesh
   * module
   * @param acoustic_free_surface Acoustic free surface boundary object defined
   * in mesh module
   */
  composite_stacey_dirichlet(
      const specfem::compute::mesh::control_nodes &control_nodes,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  int nelements; ///< Number of elements with composite boundary
  specfem::kokkos::DeviceView1d<int> ispec; ///< Element indices with composite
                                            ///< boundary
  specfem::kokkos::HostMirror1d<int> h_ispec; ///< Host mirror of ispec
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types>
      type; ///< Boundary information for each element
  specfem::kokkos::HostMirror1d<specfem::compute::access::boundary_types>
      h_type; ///< Host mirror of type
};

/**
 * @brief Struct to store all boundary types
 *
 */
struct boundaries {
  /**
   * @brief Construct a new boundaries object
   *
   * @param boundaries mesh boundaries object providing the necessary
   * information about boundaries within the mesh
   */
  boundaries(
      const specfem::compute::mesh::control_nodes &control_nodes,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);
      : acoustic_free_surface(control_nodes, absorbing_boundaries,
                              acoustic_free_surface),
        stacey(control_nodes, absorbing_boundaries, acoustic_free_surface),
        composite_stacey_dirichlet(control_nodes, absorbing_boundaries,
                              acoustic_free_surface) {}

      specfem::compute::acoustic_free_surface
          acoustic_free_surface; ///< acoustic
                                 ///< free
                                 ///< surface
                                 ///< boundary

      specfem::compute::stacey stacey; ///< Stacey boundary
      specfem::compute::composite_stacey_dirichlet
          composite_stacey_dirichlet; ///< Composite Stacey-Dirichlet boundary
};
} // namespace compute
} // namespace specfem

#endif
