#ifndef _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_TPP
#define _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_TPP

#include "compute/interface.hpp"
#include "coupled_interface/impl/edge/edge.hpp"
// #include "coupled_interface/impl/edge/elastic_acoustic/elastic_acoustic.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

// template <typename qp_type>
// specfem::coupled_interface::impl::edges::edge<
//     specfem::domain::domain<specfem::enums::element::medium::elastic,
//     qp_type>,
//     specfem::domain::domain<specfem::enums::element::medium::acoustic,
//                             qp_type> >::
//     edge(const
//     specfem::domain::domain<specfem::enums::element::medium::elastic,
//                                       qp_type> &self_domain,
//         const
//         specfem::domain::domain<specfem::enums::element::medium::acoustic,
//                                       qp_type> &coupled_domain,
//         const qp_type &quadrature_points,
//         const specfem::compute::coupled_interfaces::coupled_interfaces
//             &coupled_interfaces,
//         const specfem::compute::partial_derivatives &partial_derivatives,
//         const specfem::kokkos::DeviceView1d<type_real> wxgll,
//         const specfem::kokkos::DeviceView1d<type_real> wzgll,
//         const specfem::kokkos::DeviceView3d<int> ibool)
//     : acoustic_ispec(coupled_interfaces.elastic_acoustic.acoustic_ispec),
//       elastic_ispec(coupled_interfaces.elastic_acoustic.elastic_ispec),
//       acoustic_edge(coupled_interfaces.elastic_acoustic.acoustic_edge),
//       elastic_edge(coupled_interfaces.elastic_acoustic.elastic_edge),
//       ibool(ibool), xix(partial_derivatives.xix),
//       xiz(partial_derivatives.xiz), gammax(partial_derivatives.gammax),
//       gammaz(partial_derivatives.gammaz),
//       jacobian(partial_derivatives.jacobian),
//       self_field_dot_dot(self_domain.get_field_dot_dot()),
//       coupled_field_dot_dot(coupled_domain.get_field_dot_dot()),
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
//   assert(coupled_field_dot_dot.extent(1) == coupled_medium::components);
// #endif

//   return;
// }

// template <typename qp_type>
// KOKKOS_FUNCTION void specfem::coupled_interface::impl::edges::edge<
//     specfem::domain::domain<specfem::enums::element::medium::elastic,
//     qp_type>,
//     specfem::domain::domain<specfem::enums::element::medium::acoustic,
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
//   coupled_iterator(ipoint, acoustic_edge_type, ix, iz);

//   int iglob = ibool(acoustic_ispec_l, iz, ix);
//   type_real pressure = -1.0 * coupled_field_dot_dot(iglob, 0);

//   type_real valx, valz;

//   switch (acoustic_edge_type) {
//   case specfem::enums::edge::type::LEFT:
//     valx = -1.0 * wzgll(iz) *
//            (xix(acoustic_ispec_l, iz, ix) * jacobian(acoustic_ispec_l, iz,
//            ix) *
//             pressure);
//     valz = -1.0 * wzgll(iz) *
//            (xiz(acoustic_ispec_l, iz, ix) * jacobian(acoustic_ispec_l, iz,
//            ix) *
//             pressure);
//     self_iterator(ipoint, elastic_edge_type, ix, iz);
//     iglob = ibool(elastic_ispec_l, iz, ix);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 0), valx);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 1), valz);
//     break;
//   case specfem::enums::edge::type::RIGHT:
//     valx = wzgll(iz) * (xix(acoustic_ispec_l, iz, ix) *
//                         jacobian(acoustic_ispec_l, iz, ix) * pressure);
//     valz = wzgll(iz) * (xiz(acoustic_ispec_l, iz, ix) *
//                         jacobian(acoustic_ispec_l, iz, ix) * pressure);
//     self_iterator(ipoint, elastic_edge_type, ix, iz);
//     iglob = ibool(elastic_ispec_l, iz, ix);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 0), valx);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 1), valz);
//     break;
//   case specfem::enums::edge::type::BOTTOM:
//     valx = -1.0 * wxgll(ix) *
//            (gammax(acoustic_ispec_l, iz, ix) *
//             jacobian(acoustic_ispec_l, iz, ix) * pressure);
//     valz = -1.0 * wxgll(ix) *
//            (gammaz(acoustic_ispec_l, iz, ix) *
//             jacobian(acoustic_ispec_l, iz, ix) * pressure);
//     self_iterator(ipoint, elastic_edge_type, ix, iz);
//     iglob = ibool(elastic_ispec_l, iz, ix);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 0), valx);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 1), valz);
//     break;
//   case specfem::enums::edge::type::TOP:
//     valx = wxgll(ix) * (gammax(acoustic_ispec_l, iz, ix) *
//                         jacobian(acoustic_ispec_l, iz, ix) * pressure);
//     valz = wxgll(ix) * (gammaz(acoustic_ispec_l, iz, ix) *
//                         jacobian(acoustic_ispec_l, iz, ix) * pressure);
//     self_iterator(ipoint, elastic_edge_type, ix, iz);
//     iglob = ibool(elastic_ispec_l, iz, ix);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 0), valx);
//     Kokkos::atomic_add(&self_field_dot_dot(iglob, 1), valz);
//     break;
//   default:
//     assert(false);
//     break;
//   }
// }

template <>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic,
                                      false, false, true, false>
specfem::coupled_interface::impl::edges::edge<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>::
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

  return { specfem::datatype::ScalarPointViewType<type_real, 2>(
      factor * normal(0) * coupled_field_elements.acceleration(0),
      factor * normal(1) * coupled_field_elements.acceleration(0)) };
}

#endif // _COUPLED_INTERFACE_IMPL_ELASTIC_ACOUSTIC_EDGE_TPP
