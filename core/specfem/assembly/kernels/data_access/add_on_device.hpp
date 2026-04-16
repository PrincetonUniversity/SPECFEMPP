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
 * @brief Accumulate misfit kernel values on GPU device for seismic inversion
 *
 * @ingroup KernelsDataAccess
 *
 * This function performs point-wise accumulation of misfit kernel values.
 *
 * @tparam IndexType Index type for spectral element location
 * (specfem::point::index)
 * @tparam PointKernelType Kernel accessor type containing point-wise kernel
 * values (specfem::point::kernels)
 *
 * @param index Spatial index specifying element and quadrature point location
 * @param point_kernels Point-wise kernel values computed from forward-adjoint
 * correlation
 * @param kernels Global kernels container to accumulate values into
 *
 * @code
 * // Example: Accumulate elastic kernels during adjoint simulation
 * const specfem::point::index<...> index = ...; // Current element and GLL
 * point const specfem::point::kernels<...> point_kernels = ...;
 * const auto kernels = assembly.kernels; // Global kernels container
 * add_on_device(index, point_kernels, kernels);
 * @endcode
 */
template <typename IndexType, typename PointKernelType,
          typename std::enable_if<IndexType::using_simd ==
                                      PointKernelType::simd::using_simd,
                                  int>::type = 0>
KOKKOS_FUNCTION void
add_on_device(const IndexType &index, const PointKernelType &point_kernels,
              const kernels<PointKernelType::dimension_tag> &kernels) {

  const int ispec = kernels.property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  kernels.template get_container<MediumTag, PropertyTag>().add_device_values(
      l_index, point_kernels);

  return;
}

} // namespace specfem::assembly
