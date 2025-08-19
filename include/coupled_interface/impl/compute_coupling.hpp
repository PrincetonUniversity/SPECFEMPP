#pragma once

#include "enumerations/medium.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace coupled_interface {
namespace impl {

using elastic_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::elastic_psv>;

using acoustic_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::acoustic>;

template <typename SelfFieldType, typename CoupledFieldType,
          typename NormalViewType>
KOKKOS_INLINE_FUNCTION void
impl_compute_coupling(const acoustic_type &, const elastic_type &,
                      const type_real &factor, const NormalViewType &normal,
                      const CoupledFieldType &coupled_field,
                      SelfFieldType &self_field) {

  self_field(0) =
      factor * (normal(0) * coupled_field(0) + normal(1) * coupled_field(1));
}

template <typename SelfFieldType, typename CoupledFieldType,
          typename NormalViewType>
KOKKOS_INLINE_FUNCTION void
impl_compute_coupling(const elastic_type &, const acoustic_type &,
                      const type_real &factor, const NormalViewType &normal,
                      const CoupledFieldType &coupled_field,
                      SelfFieldType &self_field) {
  self_field(0) = factor * normal(0) * coupled_field(0);
  self_field(1) = factor * normal(1) * coupled_field(0);
}

/**
 * @brief Compute the coupling between two fields at a GLL point.
 *
 * @tparam SelfFieldType Type of the field on which the coupling is computed.
 * @tparam CoupledFieldType Type of the field coupled to the primary field.
 * @tparam NormalViewType Type of the normal vector.
 * @param factor Factor to multiply the coupling term at the GLL point.
 * @param normal Normal vector to the interface at the GLL point
 * @param coupled_field Field coupled to the primary field.
 * @param self_field Primary field.
 */
template <typename SelfFieldType, typename CoupledFieldType,
          typename NormalViewType>
KOKKOS_INLINE_FUNCTION void
compute_coupling(const type_real &factor, const NormalViewType &normal,
                 const CoupledFieldType &coupled_field,
                 SelfFieldType &self_field) {

  constexpr auto self_medium = SelfFieldType::medium_tag;
  constexpr auto coupled_medium = CoupledFieldType::medium_tag;

  using self_type =
      std::integral_constant<specfem::element::medium_tag, self_medium>;
  using coupled_type =
      std::integral_constant<specfem::element::medium_tag, coupled_medium>;

  static_assert(self_medium != coupled_medium,
                "Error: self_medium cannot be equal to coupled_medium");

  static_assert(NormalViewType::components == 2,
                "NormalViewType must have dimension 2");

  static_assert(specfem::data_access::is_point<SelfFieldType>::value &&
                    specfem::data_access::is_field<SelfFieldType>::value,
                "SelfFieldType must be a point field");

  static_assert(specfem::data_access::is_point<CoupledFieldType>::value &&
                    specfem::data_access::is_field<CoupledFieldType>::value,
                "CoupledFieldType must be a point field");

  impl_compute_coupling(self_type(), coupled_type(), factor, normal,
                        coupled_field, self_field);

  return;
}

} // namespace impl
} // namespace coupled_interface
} // namespace specfem
