#ifndef _COUPLED_INTERFACE_IMPL_ACOUSTIC_ELASTIC_TPP
#define _COUPLED_INTERFACE_IMPL_ACOUSTIC_ELASTIC_TPP

#include "compute/interface.hpp"
#include "coupled_interface/impl/edge/edge.hpp"
// #include "coupled_interface/impl/edge/elastic_acoustic/acoustic_elastic.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

// template <typename qp_type>
// specfem::coupled_interface::impl::edges::edge<
//     specfem::domain::domain<specfem::enums::element::medium::acoustic,
//     qp_type>,
//     specfem::domain::domain<specfem::enums::element::medium::elastic,
//                             qp_type> >::
//     edge(const specfem::domain::domain<
//              specfem::enums::element::medium::acoustic, qp_type>
//              &self_domain,
//          const
//          specfem::domain::domain<specfem::enums::element::medium::elastic,
//                                        qp_type> &coupled_domain,
//          const qp_type &quadrature_points,
//          const specfem::compute::coupled_interfaces::coupled_interfaces
//              &coupled_interfaces,
//          const specfem::compute::partial_derivatives &partial_derivatives,
//          const specfem::kokkos::DeviceView1d<type_real> wxgll,
//          const specfem::kokkos::DeviceView1d<type_real> wzgll,
//          const specfem::kokkos::DeviceView3d<int> ibool)
//     : acoustic_ispec(coupled_interfaces.elastic_acoustic.acoustic_ispec),
//       elastic_ispec(coupled_interfaces.elastic_acoustic.elastic_ispec),
//       acoustic_edge(coupled_interfaces.elastic_acoustic.acoustic_edge),
//       elastic_edge(coupled_interfaces.elastic_acoustic.elastic_edge),
//       ibool(ibool), xix(partial_derivatives.xix),
//       xiz(partial_derivatives.xiz), gammax(partial_derivatives.gammax),
//       gammaz(partial_derivatives.gammaz),
//       jacobian(partial_derivatives.jacobian),
//       self_field_dot_dot(self_domain.get_field_dot_dot()),
//       coupled_field(coupled_domain.get_field()),
//       quadrature_points(quadrature_points), wxgll(wxgll), wzgll(wzgll) {

//   int ngllx, ngllz;
//   quadrature_points.get_ngll(&ngllx, &ngllz);

// #ifndef NDEBUG
//   assert(ibool.extent(1) == ngllz);
//   assert(ibool.extent(2) == ngllx);
//   assert(partial_derivatives.xix.extent(1) == ngllz);
//   assert(partial_derivatives.xix.extent(2) == ngllx);
//   assert(partial_derivatives.xiz.extent(1) == ngllz);
//   assert(partial_derivatives.xiz.extent(2) == ngllx);
//   assert(partial_derivatives.gammax.extent(1) == ngllz);
//   assert(partial_derivatives.gammax.extent(2) == ngllx);
//   assert(partial_derivatives.gammaz.extent(1) == ngllz);
//   assert(partial_derivatives.gammaz.extent(2) == ngllx);
//   assert(partial_derivatives.jacobian.extent(1) == ngllz);
//   assert(partial_derivatives.jacobian.extent(2) == ngllx);
//   assert(wxgll.extent(0) == ngllx);
//   assert(wzgll.extent(0) == ngllz);
// #endif

//   self_iterator =
//       specfem::coupled_interface::impl::edges::self_iterator(ngllx, ngllz);

//   coupled_iterator =
//       specfem::coupled_interface::impl::edges::coupled_iterator(ngllx,
//       ngllz);

// #ifndef NDEBUG
//   assert(self_field_dot_dot.extent(1) == self_medium::components);
//   assert(coupled_field.extent(1) == coupled_medium::components);
// #endif

//   return;
// }

// template <typename qp_type>
// KOKKOS_FUNCTION void specfem::coupled_interface::impl::edges::edge<
//     specfem::domain::domain<specfem::enums::element::medium::acoustic,
//     qp_type>,
//     specfem::domain::domain<specfem::enums::element::medium::elastic,
//                             qp_type> >::compute_coupling(const int &iedge,
//                                                          const int &ipoint)
//     const {

//   int ngllx, ngllz;
//   quadrature_points.get_ngll(&ngllx, &ngllz);

//   const auto acoustic_edge_type = this->acoustic_edge(iedge);
//   const auto elastic_edge_type = this->elastic_edge(iedge);

//   const int acoustic_ispec_l = this->acoustic_ispec(iedge);
//   const int elastic_ispec_l = this->elastic_ispec(iedge);

//   int ix, iz;
//   coupled_iterator(ipoint, elastic_edge_type, ix, iz);

//   int iglob = ibool(elastic_ispec_l, iz, ix);
//   const type_real displ_x = coupled_field(iglob, 0);
//   const type_real displ_z = coupled_field(iglob, 1);

//   type_real val;

//   switch (acoustic_edge_type) {
//   case specfem::enums::edge::type::LEFT:
//     self_iterator(ipoint, acoustic_edge_type, ix, iz);
//     iglob = ibool(acoustic_ispec_l, iz, ix);
//     val = -1.0 * wzgll(iz) *
//           (xix(acoustic_ispec_l, iz, ix) * jacobian(acoustic_ispec_l, iz, ix)
//           *
//                displ_x +
//            xiz(acoustic_ispec_l, iz, ix) * jacobian(acoustic_ispec_l, iz, ix)
//            *
//                displ_z);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 0), val);
//     break;
//   case specfem::enums::edge::type::RIGHT:
//     self_iterator(ipoint, acoustic_edge_type, ix, iz);
//     iglob = ibool(acoustic_ispec_l, iz, ix);
//     val = wzgll(iz) * (xix(acoustic_ispec_l, iz, ix) *
//                           jacobian(acoustic_ispec_l, iz, ix) * displ_x +
//                       xiz(acoustic_ispec_l, iz, ix) *
//                           jacobian(acoustic_ispec_l, iz, ix) * displ_z);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 0), val);
//     break;
//   case specfem::enums::edge::type::BOTTOM:
//     self_iterator(ipoint, acoustic_edge_type, ix, iz);
//     iglob = ibool(acoustic_ispec_l, iz, ix);
//     val = -1.0 * wxgll(ix) *
//           (gammax(acoustic_ispec_l, iz, ix) *
//                jacobian(acoustic_ispec_l, iz, ix) * displ_x +
//            gammaz(acoustic_ispec_l, iz, ix) *
//                jacobian(acoustic_ispec_l, iz, ix) * displ_z);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 0), val);
//     break;
//   case specfem::enums::edge::type::TOP:
//     self_iterator(ipoint, acoustic_edge_type, ix, iz);
//     iglob = ibool(acoustic_ispec_l, iz, ix);
//     val = wxgll(ix) * (gammax(acoustic_ispec_l, iz, ix) *
//                           jacobian(acoustic_ispec_l, iz, ix) * displ_x +
//                       gammaz(acoustic_ispec_l, iz, ix) *
//                           jacobian(acoustic_ispec_l, iz, ix) * displ_z);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 0), val);
//     break;
//   default:
//     break;
//   }

//   return;
// }

template <>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::acoustic,
                                      false, false, true, false>
specfem::coupled_interface::impl::edges::edge<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic>::
    compute_coupling_terms(
        const specfem::datatype::ScalarPointViewType<type_real, 2> &normal,
        const specfem::datatype::ScalarPointViewType<type_real, 2> &weights,
        const specfem::edge::interface &coupled_edge,
        const CoupledPointFieldType &coupled_field_elements) const {

  const type_real factor = [&]() -> type_real {
    switch (coupled_edge.type) {
    case specfem::enums::edge::type::LEFT:
    case specfem::enums::edge::type::RIGHT:
      return -1.0 * weights(1);
      break;
    case specfem::enums::edge::type::BOTTOM:
    case specfem::enums::edge::type::TOP:
      return -1.0 * weights(0);
      break;
    default:
      DEVICE_ASSERT(false, "Invalid edge type");
      return 0.0;
      break;
    }
  }();

  return { specfem::datatype::ScalarPointViewType<type_real, 1>(
      factor * (normal(0) * coupled_field_elements.displacement(0) +
                normal(1) * coupled_field_elements.displacement(1))) };
}

#endif // _COUPLED_INTERFACE_IMPL_ACOUSTIC_ELASTIC_TPP
