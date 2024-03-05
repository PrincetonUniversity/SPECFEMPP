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
 * @brief Coupling edge class to define coupling physics between 2 domains.
 *
 * @tparam self_domain Primary domain of the interface.
 * @tparam coupled_domain Coupled domain of the interface.
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag SelfMedium,
          specfem::element::medium_tag CoupledMedium>
class edge {

public:
  using self_medium_type = specfem::medium::medium<DimensionType, SelfMedium>;
  using coupled_medium_type =
      specfem::medium::medium<DimensionType, CoupledMedium>;

  edge(){};

  edge(const specfem::compute::assembly &assembly){};

  KOKKOS_FUNCTION
  specfem::kokkos::array_type<type_real, self_medium_type::components>
  compute_coupling_terms(
      const specfem::kokkos::array_type<type_real, 2> &normal,
      const specfem::kokkos::array_type<type_real, 2> &weights,
      const specfem::edge::interface &coupled_edge_type,
      const specfem::kokkos::array_type<
          type_real, coupled_medium_type::components> &field) const;

  KOKKOS_FUNCTION
  specfem::kokkos::array_type<type_real, coupled_medium_type::components>
  load_field_elements(
      const int global_index,
      const specfem::compute::impl::field_impl<coupled_medium_type> &field)
      const;
};
} // namespace edges
} // namespace impl
} // namespace coupled_interface
} // namespace specfem

#endif // _COUPLED_INTERFACE_IMPL_EDGE_HPP
