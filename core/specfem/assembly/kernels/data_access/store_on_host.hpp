#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "specfem/assembly/kernels.hpp"

namespace specfem::assembly {

/**
 * @defgroup KernelsDataAccess Misfit Kernels Data Access Functions
 *
 */

/**
 * @brief Store misfit kernel values on CPU host for seismic inversion
 *
 * This function performs point-wise storage of misfit kernel values.
 *
 * @ingroup KernelsDataAccess
 *
 * @tparam PointKernelType Kernel accessor type containing point-wise kernel
 * values (specfem::point::kernels)
 * @tparam IndexType Index type for spectral element location
 * (specfem::point::index)
 *
 * @param index Spatial index specifying element and quadrature point location
 * @param point_kernels Point-wise kernel values to store into global container
 * @param kernels Global kernels container to store values into
 *
 * @code
 * // Example: Store elastic kernels during adjoint simulation
 * const specfem::point::index<...> index = ...; // Current element and GLL
 * point const specfem::point::kernels<...> point_kernels = ...; const auto
 * kernels = assembly.kernels; // Global kernels container store_on_host(index,
 * point_kernels, kernels);
 * @endcode
 */
template <typename PointKernelType, typename IndexType,
          typename std::enable_if<IndexType::using_simd ==
                                      PointKernelType::simd::using_simd,
                                  int>::type = 0>
void store_on_host(const IndexType &index, const PointKernelType &point_kernels,
                   const kernels<PointKernelType::dimension_tag> &kernels) {
  const int ispec = kernels.h_property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  kernels.template get_container<MediumTag, PropertyTag>().store_host_values(
      l_index, point_kernels);

  return;
}

} // namespace specfem::assembly
