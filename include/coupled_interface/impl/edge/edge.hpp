#ifndef _COUPLED_INTERFACE_IMPL_EDGE_HPP
#define _COUPLED_INTERFACE_IMPL_EDGE_HPP

#include "compute/coupled_interfaces.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace coupled_interface {
namespace impl {
namespace edges {

/**
 * @brief Wrapper for iterator class to iterate through points on the primary
 * domain of a coupled interface.
 *
 */
struct self_iterator {
  int ngllx; ///< Number of GLL points in the x direction.
  int ngllz; ///< Number of GLL points in the z direction.

  self_iterator() = default;

  /**
   * @brief Construct a new self iterator object
   *
   * @param ngllx ///< Number of GLL points in the x direction.
   * @param ngllz ///< Number of GLL points in the z direction.
   */
  self_iterator(const int &ngllx, const int &ngllz)
      : ngllx(ngllx), ngllz(ngllz){};

  /**
   * @brief Operator to iterate through points on the primary domain of a
   * coupled interface.
   *
   * @param ipoint Index of the point on the edge of the primary domain.
   * @param iedge_type Type of the edge of the primary domain.
   * @param i X index of the quadrature point inside the element of the primary
   * domain which forms the edge.
   * @param j Z index of the quadrature point inside the element of the coupled
   * domain which forms the edge.
   */
  KOKKOS_FUNCTION
  void operator()(const int &ipoint,
                  const specfem::enums::coupling::edge::type &iedge_type,
                  int &i, int &j) const {
    specfem::compute::coupled_interfaces::iterator::self_iterator(
        ipoint, iedge_type, this->ngllx, this->ngllz, i, j);
    return;
  }
};

/**
 * @brief Wrapper for iterator class to iterate through points on the coupled
 * domain of a coupled interface.
 *
 */
struct coupled_iterator {
  int ngllx; ///< Number of GLL points in the x direction.
  int ngllz; ///< Number of GLL points in the z direction.

  coupled_iterator() = default;

  /**
   * @brief Construct a new coupled iterator object
   *
   * @param ngllx Number of GLL points in the x direction.
   * @param ngllz Number of GLL points in the z direction.
   */
  coupled_iterator(const int &ngllx, const int &ngllz)
      : ngllx(ngllx), ngllz(ngllz){};

  /**
   * @brief Operator to iterate through points on the coupled domain of a
   * coupled interface.
   *
   * @param ipoint Index of the point on the edge of the coupled domain.
   * @param iedge_type Type of the edge of the coupled domain.
   * @param i X index of the quadrature point inside the element of the coupled
   * domain which forms the edge.
   * @param j Z index of the quadrature point inside the element of the coupled
   * domain which forms the edge.
   */
  KOKKOS_FUNCTION
  void operator()(const int &ipoint,
                  const specfem::enums::coupling::edge::type &iedge_type,
                  int &i, int &j) const {
    specfem::compute::coupled_interfaces::iterator::coupled_iterator(
        ipoint, iedge_type, this->ngllx, this->ngllz, i, j);
    return;
  }
};

/**
 * @brief Coupling edge class to define coupling physics between 2 domains.
 *
 * @tparam self_domain Primary domain of the interface.
 * @tparam coupled_domain Coupled domain of the interface.
 */
template <class self_domain, class coupled_domain> class edge {};
} // namespace edges
} // namespace impl
} // namespace coupled_interface
} // namespace specfem

#endif // _COUPLED_INTERFACE_IMPL_EDGE_HPP
