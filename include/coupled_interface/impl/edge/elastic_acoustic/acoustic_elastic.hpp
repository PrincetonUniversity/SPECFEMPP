#ifndef _COUPLED_INTERFACE_IMPL_ACOUSTIC_ELASTIC_EDGE_HPP
#define _COUPLED_INTERFACE_IMPL_ACOUSTIC_ELASTIC_EDGE_HPP

#include "coupled_interface/impl/edge/edge.hpp"
#include "domain/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace coupled_interface {
namespace impl {
namespace edges {
/**
 * @brief Template specialization for the edge between an acoustic and an
 * elastic domain.
 *
 * @tparam qp_type Quadrature points type.
 */
template <>
class edge<specfem::enums::element::medium::acoustic,
           specfem::enums::element::medium::elastic> {
public:
  using self_medium_type = specfem::enums::element::medium::acoustic;
  using coupled_medium_type = specfem::enums::element::medium::elastic;

  edge(){};

  /**
   * @brief Construct a new coupling edge object
   *
   * @param self_domain Primary domain of the interface (acoustic).
   * @param coupled_domain Coupled domain of the interface (elastic).
   * @param quadrature_points A quadrature points object defining the quadrature
   * points either at compile time or run time.
   * @param coupled_interfaces struct containing the coupling information.
   * @param partial_derivatives struct containing the partial derivatives.
   * @param wxgll weights for the GLL quadrature points in the x direction.
   * @param wzgll weights for the GLL quadrature points in the z direction.
   * @param ibool Global indexing for all GLL points
   */
  edge(const specfem::compute::assembly &assembly);

  /**
   * @brief Compute coupling interaction between domains
   *
   * @param iedge Index of the edge.
   * @param ipoint Index of the quadrature point on the edge.
   */
  KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, 1>
  specfem::coupled_interface::impl::edges::edge<
      specfem::enums::element::medium::elastic,
      specfem::enums::element::medium::acoustic>::
      compute_coupling_terms(
          const specfem::kokkos::array_type<type_real, 2> &normal,
          const specfem::kokkos::array_type<type_real, 2> &weights,
          const specfem::enums::edge::type &coupled_edge_type,
          const specfem::kokkos::array_type<type_real, 2> &pressure) const;

  KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, 2>
  specfem::coupled_interface::impl::edges::edge<
      specfem::enums::element::medium::elastic,
      specfem::enums::element::medium::acoustic>::
      load_field_elements(
          const int coupled_global_index,
          const specfem::compute::impl::field_impl<coupled_medium_type>
              &coupled_field) const;

private:
  //   specfem::kokkos::DeviceView1d<int> acoustic_ispec; ///< Index of acoustic
  //                                                      ///< elements on the
  //                                                      edge
  //   specfem::kokkos::DeviceView1d<int> elastic_ispec;  ///< Index of elastic
  //                                                      ///< elements on the
  //                                                      edge
  //   specfem::kokkos::DeviceView3d<int> ibool;     ///< Global indexing for
  //   all GLL
  //                                                 ///< points
  //   specfem::kokkos::DeviceView3d<type_real> xix; ///< xix
  //   specfem::kokkos::DeviceView3d<type_real> xiz; ///< xiz
  //   specfem::kokkos::DeviceView3d<type_real> gammax;   ///< gammax
  //   specfem::kokkos::DeviceView3d<type_real> gammaz;   ///< gammaz
  //   specfem::kokkos::DeviceView3d<type_real> jacobian; ///< Jacobian
  //   specfem::kokkos::DeviceView1d<specfem::enums::edge::type>
  //       acoustic_edge; ///< Orientation of edges on the acoustic domain
  //   specfem::kokkos::DeviceView1d<specfem::enums::edge::type>
  //       elastic_edge; ///< Orientation of edges on the elastic domain
  //   specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  //       self_field_dot_dot; ///< Second derivative of potential field on the
  //                           ///< acoustic domain
  //   specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  //       coupled_field;         ///< Displacement on the elastic domain
  //   qp_type quadrature_points; ///< Quadrature points object defining the
  //                              ///< quadrature points either at compile time
  //                              or
  //                              ///< run time.
  //   specfem::kokkos::DeviceView1d<type_real> wxgll; ///< Weights for the GLL
  //                                                   ///< quadrature points in
  //                                                   the
  //                                                   ///< x direction
  //   specfem::kokkos::DeviceView1d<type_real> wzgll; ///< Weights for the GLL
  //                                                   ///< quadrature points in
  //                                                   the
  //                                                   ///< z direction

  //   specfem::coupled_interface::impl::edges::self_iterator
  //       self_iterator; ///< Iterator for points on the edge in the primary
  //       domain
  //   specfem::coupled_interface::impl::edges::coupled_iterator
  //       coupled_iterator; ///< Iterator for points on the edge in the coupled
  //                         ///< domain
};
} // namespace edges
} // namespace impl
} // namespace coupled_interface
} // namespace specfem

#endif // _COUPLED_INTERFACE_IMPL_ACOUSTIC_ELASTIC_EDGE_HPP
