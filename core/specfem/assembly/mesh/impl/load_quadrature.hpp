#pragma once

#include "enumerations/interface.hpp"
#include "specfem/execution.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
/**
 * @defgroup MeshDataAccess
 *
 */

/**
 * @brief Load quadrature data for a spectral element on host or device
 *
 * @ingroup MeshDataAccess
 *
 * @tparam MemberType Member type. Needs to be a Kokkos::TeamPolicy member type
 * @tparam ViewType View type. Needs to be of @ref
 * specfem::quadrature::lagrange_derivative
 * @param team Team member
 * @param mesh Mesh data
 * @param lagrange_derivative Quadrature data for the element (output)
 */
template <bool on_device, typename MemberType, typename ViewType>
KOKKOS_INLINE_FUNCTION void impl_load(
    const MemberType &team,
    const specfem::assembly::mesh_impl::quadrature<ViewType::dimension_tag>
        &quadrature,
    ViewType &lagrange_derivative) {

  constexpr int NGLL = ViewType::ngll;

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a host execution space");

  specfem::execution::for_each_level(
      specfem::execution::TeamThreadMDRangeIterator(team, NGLL, NGLL),
      [&](const auto index) {
        int iz = index(0);
        int ix = index(1);
        if constexpr (on_device) {
          lagrange_derivative.hprime_gll(iz, ix) = quadrature.hprime(iz, ix);
        } else {
          lagrange_derivative.hprime_gll(iz, ix) = quadrature.h_hprime(iz, ix);
        }
      });
}

/**
 * @defgroup MeshDataAccess
 *
 */

/**
 * @brief Load quadrature data for a spectral element on the device
 *
 * @ingroup MeshDataAccess
 *
 * @tparam MemberType Member type. Needs to be a Kokkos::TeamPolicy member type
 * @tparam ViewType View type. Needs to be of @ref
 * specfem::quadrature::lagrange_derivative
 * @param team Team member
 * @param mesh Mesh data
 * @param lagrange_derivative Quadrature data for the element (output)
 */
template <typename MemberType, typename ViewType,
          typename std::enable_if_t<
              (ViewType::data_class_type ==
               specfem::data_access::DataClassType::lagrange_derivative),
              int> = 0>
KOKKOS_FUNCTION void load_on_device(
    const MemberType &team,
    const specfem::assembly::mesh_impl::quadrature<ViewType::dimension_tag>
        &quadrature,
    ViewType &lagrange_derivative) {

  impl_load<true>(team, quadrature, lagrange_derivative);
}

/**
 * @brief Load quadrature data for a spectral element on the host
 *
 * @ingroup MeshDataAccess
 *
 * @tparam MemberType Member type. Needs to be a Kokkos::TeamPolicy member type
 * @tparam ViewType View type. Needs to be of @ref
 * specfem::quadrature::lagrange_derivative
 * @param team Team member
 * @param mesh Mesh data
 * @param lagrange_derivative Quadrature data for the element (output)
 */
template <typename MemberType, typename ViewType,
          typename std::enable_if_t<
              (ViewType::data_class_type ==
               specfem::data_access::DataClassType::lagrange_derivative),
              int> = 0>
void load_on_host(
    const MemberType &team,
    const specfem::assembly::mesh_impl::quadrature<ViewType::dimension_tag>
        &quadrature,
    ViewType &lagrange_derivative) {
  impl_load<false>(team, quadrature, lagrange_derivative);
}
} // namespace specfem::assembly
