#ifndef _COUPLED_INTERFACE_HPP_
#define _COUPLED_INTERFACE_HPP_

#include "compute/interface.hpp"
#include "impl/edge/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace coupled_interface {
template <class self_domain_type, class coupled_domain_type>
class coupled_interface {
public:
  using self_medium = typename self_domain_type::medium_type;
  using coupled_medium = typename coupled_domain_type::medium_type;
  using quadrature_points_type =
      typename self_domain_type::quadrature_points_type;

  coupled_interface(
      self_domain_type &self_domain, coupled_domain_type &coupled_domain,
      const specfem::compute::coupled_interfaces::coupled_interfaces
          &coupled_interfaces,
      const quadrature_points_type &quadrature_points,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::kokkos::DeviceView3d<int> ibool,
      const specfem::kokkos::DeviceView1d<type_real> wxgll,
      const specfem::kokkos::DeviceView1d<type_real> wzgll);
  void compute_coupling();

private:
  self_domain_type self_domain;
  coupled_domain_type coupled_domain;
  quadrature_points_type quadrature_points;
  specfem::kokkos::DeviceView1d<specfem::coupled_interface::impl::edges::edge<
      self_domain_type, coupled_domain_type> >
      edges;
};
} // namespace coupled_interface
} // namespace specfem
#endif // _COUPLED_INTERFACES_HPP_
