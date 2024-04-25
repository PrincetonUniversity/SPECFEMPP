#ifndef _COUPLED_INTERFACE_IMPL_EDGE_HPP
#define _COUPLED_INTERFACE_IMPL_EDGE_HPP

#include "compute/coupled_interfaces.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace coupled_interface {
namespace impl {
namespace edges {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag SelfMedium,
          specfem::element::medium_tag CoupledMedium>
class edge_impl;

template <>
class edge_impl<specfem::dimension::type::dim2,
                specfem::element::medium_tag::acoustic,
                specfem::element::medium_tag::elastic> {

public:
  using self_medium_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::acoustic>;
  using coupled_medium_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::elastic>;

  using CoupledPointFieldType =
      specfem::point::field<coupled_medium_type::dimension,
                            coupled_medium_type::medium_tag, true, false, false,
                            false>;
  using SelfPointFieldType = specfem::point::field<self_medium_type::dimension,
                                                   self_medium_type::medium_tag,
                                                   false, false, true, false>;
};

template <>
class edge_impl<specfem::dimension::type::dim2,
                specfem::element::medium_tag::elastic,
                specfem::element::medium_tag::acoustic> {
public:
  using self_medium_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::elastic>;
  using coupled_medium_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::acoustic>;

  using CoupledPointFieldType =
      specfem::point::field<coupled_medium_type::dimension,
                            coupled_medium_type::medium_tag, false, false, true,
                            false>;

  using SelfPointFieldType = specfem::point::field<self_medium_type::dimension,
                                                   self_medium_type::medium_tag,
                                                   false, false, true, false>;
};

/**
 * @brief Coupling edge class to define coupling physics between 2 domains.
 *
 * @tparam self_domain Primary domain of the interface.
 * @tparam coupled_domain Coupled domain of the interface.
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag SelfMedium,
          specfem::element::medium_tag CoupledMedium>
class edge : public edge_impl<DimensionType, SelfMedium, CoupledMedium> {

public:
  using SelfPointFieldType =
      typename edge_impl<DimensionType, SelfMedium,
                         CoupledMedium>::SelfPointFieldType;
  using CoupledPointFieldType =
      typename edge_impl<DimensionType, SelfMedium,
                         CoupledMedium>::CoupledPointFieldType;
  edge(){};

  edge(const specfem::compute::assembly &assembly){};

  KOKKOS_FUNCTION
  SelfPointFieldType compute_coupling_terms(
      const specfem::kokkos::array_type<type_real, 2> &normal,
      const specfem::kokkos::array_type<type_real, 2> &weights,
      const specfem::edge::interface &coupled_edge_type,
      const CoupledPointFieldType &field) const;

  template <specfem::wavefield::type WaveFieldType>
  KOKKOS_FUNCTION CoupledPointFieldType load_field_elements(
      const specfem::point::index &index,
      const specfem::compute::simulation_field<WaveFieldType> &field) const {
    CoupledPointFieldType field_elements;
    specfem::compute::load_on_device(index, field, field_elements);
    return field_elements;
  }
};
} // namespace edges
} // namespace impl
} // namespace coupled_interface
} // namespace specfem

#endif // _COUPLED_INTERFACE_IMPL_EDGE_HPP
