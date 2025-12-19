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
 * @brief Load misfit kernel values on GPU device for seismic inversion
 *
 * This function performs point-wise loading of misfit kernel values.
 *
 *
 * @ingroup KernelsDataAccess
 *
 * @tparam PointKernelType Kernel accessor type containing point-wise kernel
 * values (specfem::point::kernels)
 * @tparam IndexType Index type for spectral element location
 * (specfem::point::index)
 *
 * @param index Spatial index specifying element and quadrature point location
 * @param kernels Global kernels container to load values from
 * @param point_kernels Point-wise kernel values loaded from global container
 *
 * @code
 * // Example: Load elastic kernels during adjoint simulation
 * const auto index = specfem::point::index<...>  ...; // Current element and
 * GLL point specfem::point::kernels<...> point_kernels = ...; const auto
 * kernels = assembly.kernels; // Global kernels container load_on_device(index,
 * kernels, point_kernels);
 * @endcode
 */
template <typename PointKernelType, typename IndexType,
          typename std::enable_if<IndexType::using_simd ==
                                      PointKernelType::simd::using_simd,
                                  int>::type = 0>
KOKKOS_FUNCTION void
load_on_device(const IndexType &index,
               const kernels<PointKernelType::dimension_tag> &kernels,
               PointKernelType &point_kernels) {
  const int ispec = kernels.property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  kernels.template get_container<MediumTag, PropertyTag>().load_device_values(
      l_index, point_kernels);

  return;
}

} // namespace specfem::assembly
