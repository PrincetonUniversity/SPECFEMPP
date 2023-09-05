#ifndef _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_HPP
#define _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_HPP

#include "coupled_interface/impl/edge/edge.hpp"
#include "domain/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace coupled_interface {
namespace impl {

template <typename qp_type>
class edge<
    specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>,
    specfem::domain::domain<specfem::enums::element::medium::acoustic,
                            qp_type> > {

public:
  using self_medium =
      typename specfem::domain::domain<specfem::enums::element::medium::elastic,
                                       qp_type>::medium_type;
  using coupled_medium = typename specfem::domain::domain<
      specfem::enums::element::medium::acoustic, qp_type>::medium_type;
  using quadrature_points_type = qp_type;

  edge(){};

  edge(const int &inum_edge,
       const specfem::domain::domain<specfem::enums::element::medium::elastic,
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

  void compute_coupling(const int &ipoint) const;

  void
  get_edges(specfem::enums::coupling::edge::type &self_edge_type,
            specfem::enums::coupling::edge::type &coupled_edge_type) const {
    self_edge_type = this->self_edge_type;
    coupled_edge_type = this->coupled_edge_type;
    return;
  }

private:
  specfem::kokkos::DeviceView2d<int> self_ibool;
  specfem::kokkos::DeviceView2d<int> coupled_ibool;
  specfem::kokkos::DeviceView2d<type_real> xix;
  specfem::kokkos::DeviceView2d<type_real> xiz;
  specfem::kokkos::DeviceView2d<type_real> gammax;
  specfem::kokkos::DeviceView2d<type_real> gammaz;
  specfem::kokkos::DeviceView2d<type_real> jacobian;
  specfem::enums::coupling::edge::type self_edge_type;
  specfem::enums::coupling::edge::type coupled_edge_type;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
      self_field_dot_dot;
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> coupled_field;
  qp_type quadrature_points;
  specfem::kokkos::DeviceView1d<type_real> wxgll;
  specfem::kokkos::DeviceView1d<type_real> wzgll;
};

} // namespace impl
} // namespace coupled_interface
} // namespace specfem

#endif /* _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_HPP */
