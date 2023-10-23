#ifndef _COUPLED_INTERFACE_HPP_
#define _COUPLED_INTERFACE_HPP_

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "impl/edge/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"

namespace specfem {
namespace coupled_interface {
/**
 * @brief Class to compute the coupling between two domains.
 *
 * @tparam self_domain_type Primary domain of the interface.
 * @tparam coupled_domain_type Coupled domain of the interface.
 */
template <class self_domain_type, class coupled_domain_type>
class coupled_interface {
public:
  /**
   * @brief Typedefs
   *
   */
  ///@{
  /**
   * @brief Self medium type.
   *
   */
  using self_medium = typename self_domain_type::medium_type;
  /**
   * @brief Coupled medium type.
   *
   */
  using coupled_medium = typename coupled_domain_type::medium_type;
  /**
   * @brief Quadrature points object to define the quadrature points either at
   * compile time or run time.
   *
   */
  using quadrature_points_type =
      typename self_domain_type::quadrature_points_type;
  ///@}

  /**
   * @brief Construct a new coupled interface object
   *
   * @param self_domain Primary domain of the interface.
   * @param coupled_domain Coupled domain of the interface.
   * @param coupled_interfaces struct containing the coupling information.
   * @param quadrature_points A quadrature points object defining the quadrature
   * points either at compile time or run time.
   * @param partial_derivatives struct containing the partial derivatives.
   * @param ibool Global index of the GLL points.
   * @param wxgll weights for the GLL quadrature points in the x direction.
   * @param wzgll weights for the GLL quadrature points in the z direction.
   */
  coupled_interface(
      self_domain_type &self_domain, coupled_domain_type &coupled_domain,
      const specfem::compute::coupled_interfaces::coupled_interfaces
          &coupled_interfaces,
      const quadrature_points_type &quadrature_points,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::kokkos::DeviceView3d<int> ibool,
      const specfem::kokkos::DeviceView1d<type_real> wxgll,
      const specfem::kokkos::DeviceView1d<type_real> wzgll);

  /**
   * @brief Compute the coupling between the primary and coupled domains.
   *
   */
  void compute_coupling();

private:
  int nedges; ///< Number of edges in the interface.
  specfem::kokkos::DeviceView1d<specfem::enums::coupling::edge::type>
      self_edge; ///< Orientation of the edge of the primary domain.
  specfem::kokkos::DeviceView1d<specfem::enums::coupling::edge::type>
      coupled_edge; ///< Orientation of the edge of the coupled domain.
  self_domain_type self_domain;       ///< Primary domain of the interface.
  coupled_domain_type coupled_domain; ///< Coupled domain of the interface.
  quadrature_points_type quadrature_points; ///< Quadrature points object to
                                            ///< define the quadrature points
                                            ///< either at compile time or run
                                            ///< time.
  specfem::coupled_interface::impl::edges::edge<self_domain_type,
                                                coupled_domain_type>
      edge; ///< Edge class to implement coupling physics
};
} // namespace coupled_interface
} // namespace specfem
#endif // _COUPLED_INTERFACES_HPP_
