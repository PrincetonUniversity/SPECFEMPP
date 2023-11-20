#ifndef _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_HPP
#define _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_HPP

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
template <typename qp_type>
class edge<
    specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>,
    specfem::domain::domain<specfem::enums::element::medium::acoustic,
                            qp_type> > {

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  /**
   * @brief Self medium type.
   */
  using self_medium =
      typename specfem::domain::domain<specfem::enums::element::medium::elastic,
                                       qp_type>::medium_type;
  /**
   * @brief Coupled medium type.
   *
   */
  using coupled_medium = typename specfem::domain::domain<
      specfem::enums::element::medium::acoustic, qp_type>::medium_type;
  /**
   * @brief Quadrature points type.
   *
   */
  using quadrature_points_type = qp_type;
  ///@}

  edge(){};

  /**
   * @brief Construct a new coupling edge object
   *
   * @param self_domain Primary domain of the interface (elastic).
   * @param coupled_domain Coupled domain of the interface (acoustic).
   * @param quadrature_points A quadrature points object defining the quadrature
   * points either at compile time or run time.
   * @param coupled_interfaces struct used to store coupling information.
   * @param partial_derivatives struct used to store partial derivatives.
   * @param wxgll weights for the GLL quadrature points in the x direction.
   * @param wzgll weights for the GLL quadrature points in the z direction.
   * @param ibool Global indexing for all GLL points
   */
  edge(const specfem::domain::domain<specfem::enums::element::medium::elastic,
                                     qp_type> &self_domain,
       const specfem::domain::domain<specfem::enums::element::medium::acoustic,
                                     qp_type> &coupled_domain,
       const qp_type &quadrature_points,
       const specfem::compute::coupled_interfaces::coupled_interfaces
           &coupled_interfaces,
       const specfem::compute::partial_derivatives &partial_derivatives,
       const specfem::kokkos::DeviceView1d<type_real> wxgll,
       const specfem::kokkos::DeviceView1d<type_real> wzgll,
       const specfem::kokkos::DeviceView3d<int> ibool);

  /**
   * @brief Compute coupling interaction between domains
   *
   * @param iedge Index of the edge.
   * @param ipoint Index of the quadrature point on the edge.
   */
  KOKKOS_FUNCTION
  void compute_coupling(const int &iedge, const int &ipoint) const;

  /**
   * @brief Get the orientation of coupling edges
   *
   * @param iedge Index of the edge.
   * @param self_edge_type Orientation of edge in primary domain.
   * @param coupled_edge_type Orientation of edge in coupled domain.
   */
  KOKKOS_FUNCTION void
  get_edges(const int &iedge, specfem::enums::edge::type &self_edge_type,
            specfem::enums::edge::type &coupled_edge_type) const {
    self_edge_type = this->elastic_edge(iedge);
    coupled_edge_type = this->acoustic_edge(iedge);
    return;
  }

private:
  specfem::kokkos::DeviceView1d<int> acoustic_ispec; ///< Index of acoustic
                                                     ///< elements on the edge
  specfem::kokkos::DeviceView1d<int> elastic_ispec;  ///< Index of elastic
                                                     ///< elements on the edge
  specfem::kokkos::DeviceView3d<int> ibool;     ///< Global indexing for all GLL
                                                ///< points
  specfem::kokkos::DeviceView3d<type_real> xix; ///< xix
  specfem::kokkos::DeviceView3d<type_real> xiz; ///< xiz
  specfem::kokkos::DeviceView3d<type_real> gammax;   ///< gammax
  specfem::kokkos::DeviceView3d<type_real> gammaz;   ///< gammaz
  specfem::kokkos::DeviceView3d<type_real> jacobian; ///< Jacobian
  specfem::kokkos::DeviceView1d<specfem::enums::edge::type>
      acoustic_edge; ///< Orientation of edge in acoustic domain
  specfem::kokkos::DeviceView1d<specfem::enums::edge::type>
      elastic_edge; ///< Orientation of edge in elastic domain
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
      self_field_dot_dot; ///< Acceleration in elastic domain
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
      coupled_field_dot_dot; ///< Second derivative of potential in acoustic
                             ///< domain
  qp_type quadrature_points; ///< Quadrature points object to define quadrature
                             ///< points at compile time or run time
  specfem::kokkos::DeviceView1d<type_real> wxgll; ///< Weights for the GLL
                                                  ///< quadrature points in the
                                                  ///< x direction
  specfem::kokkos::DeviceView1d<type_real> wzgll; ///< Weights for the GLL
                                                  ///< quadrature points in the
                                                  ///< z direction

  specfem::coupled_interface::impl::edges::self_iterator
      self_iterator; ///< Iterator for points on the edge in the primary domain
  specfem::coupled_interface::impl::edges::coupled_iterator
      coupled_iterator; ///< Iterator for points on the edge in the coupled
                        ///< domain
};
} // namespace edges
} // namespace impl
} // namespace coupled_interface
} // namespace specfem

#endif /* _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_HPP */
