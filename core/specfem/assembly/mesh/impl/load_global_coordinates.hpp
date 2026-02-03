#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @defgroup MeshDataAccess
 *
 */

//------------------------------------------------------------------------------
// 2D Implementation
//------------------------------------------------------------------------------

/**
 * @brief Load global coordinates for a 2D GLL point (implementation)
 *
 * @ingroup MeshDataAccess
 *
 * @tparam on_device Whether to load from device or host memory
 * @tparam IndexType Index type. Must be a point index type
 * @tparam ContainerType Container type. Must have global_coordinates data_class
 * @tparam PointType Point type. Must be global_coordinates
 * @param index GLL point index
 * @param container Points container with coordinate data
 * @param point Global coordinates point structure (output)
 */
template <
    bool on_device, typename IndexType, typename ContainerType,
    typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            IndexType::dimension_tag == specfem::dimension::type::dim2 &&
            specfem::data_access::is_global_coordinates<ContainerType>::value &&
            specfem::data_access::is_global_coordinates<PointType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(const IndexType &index,
                                           const ContainerType &container,
                                           PointType &point) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  if constexpr (on_device) {
    point.x = container.coord(0, ispec, iz, ix);
    point.z = container.coord(1, ispec, iz, ix);
  } else {
    point.x = container.h_coord(0, ispec, iz, ix);
    point.z = container.h_coord(1, ispec, iz, ix);
  }
}

//------------------------------------------------------------------------------
// 3D Implementation
//------------------------------------------------------------------------------

/**
 * @brief Load global coordinates for a 3D GLL point (implementation)
 *
 * @ingroup MeshDataAccess
 *
 * @tparam on_device Whether to load from device or host memory
 * @tparam IndexType Index type. Must be a point index type
 * @tparam ContainerType Container type. Must have global_coordinates data_class
 * @tparam PointType Point type. Must be global_coordinates
 * @param index GLL point index
 * @param container Points container with coordinate data
 * @param point Global coordinates point structure (output)
 */
template <
    bool on_device, typename IndexType, typename ContainerType,
    typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            IndexType::dimension_tag == specfem::dimension::type::dim3 &&
            specfem::data_access::is_global_coordinates<ContainerType>::value &&
            specfem::data_access::is_global_coordinates<PointType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(const IndexType &index,
                                           const ContainerType &container,
                                           PointType &point) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int iy = index.iy;
  const int ix = index.ix;

  if constexpr (on_device) {
    point.x = container.coord(ispec, iz, iy, ix, 0);
    point.y = container.coord(ispec, iz, iy, ix, 1);
    point.z = container.coord(ispec, iz, iy, ix, 2);
  } else {
    point.x = container.h_coord(ispec, iz, iy, ix, 0);
    point.y = container.h_coord(ispec, iz, iy, ix, 1);
    point.z = container.h_coord(ispec, iz, iy, ix, 2);
  }
}

//------------------------------------------------------------------------------
// Public API: load_on_device
//------------------------------------------------------------------------------

/**
 * @brief Load global coordinates for a GLL point from device memory
 *
 * @ingroup MeshDataAccess
 *
 * This function transfers global coordinate data (x, z for 2D; x, y, z for 3D)
 * from a mesh points container to a local point structure for GPU-based
 * computations.
 *
 * @tparam IndexType Point index type (@ref specfem::point::index)
 * @tparam ContainerType Points container type
 * @tparam PointType Local point global_coordinates type (@ref
 * specfem::point::global_coordinates)
 *
 * @param index Quadrature point indices (ispec, iz, [iy,] ix)
 * @param container Global points container
 * @param point Local point global coordinates structure (output)
 *
 * @pre IndexType must satisfy is_index_type constraint
 * @pre ContainerType and PointType must satisfy is_global_coordinates
 * constraint
 *
 * @code
 * // Example usage in GPU kernel
 * specfem::point::index<specfem::dimension::type::dim2> idx(ispec, iz, ix);
 * specfem::point::global_coordinates<specfem::dimension::type::dim2>
 * point_coord; specfem::assembly::load_on_device(idx, mesh.points, point_coord);
 *
 * // Access loaded values
 * type_real x_val = point_coord.x;
 * type_real z_val = point_coord.z;
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            specfem::data_access::is_global_coordinates<ContainerType>::value &&
            specfem::data_access::is_global_coordinates<PointType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const ContainerType &container,
                                                PointType &point) {
  static_assert(IndexType::dimension_tag == ContainerType::dimension_tag,
                "Index and container dimension tags must match");
  static_assert(IndexType::dimension_tag == PointType::dimension_tag,
                "Index and point dimension tags must match");

  impl_load<true>(index, container, point);
}

//------------------------------------------------------------------------------
// Public API: load_on_host
//------------------------------------------------------------------------------

/**
 * @brief Load global coordinates for a GLL point from host memory
 *
 * @ingroup MeshDataAccess
 *
 * This function transfers global coordinate data (x, z for 2D; x, y, z for 3D)
 * from a mesh points container to a local point structure for host-based
 * computations.
 *
 * @tparam IndexType Point index type (@ref specfem::point::index)
 * @tparam ContainerType Points container type
 * @tparam PointType Local point global_coordinates type (@ref
 * specfem::point::global_coordinates)
 *
 * @param index Quadrature point indices (ispec, iz, [iy,] ix)
 * @param container Global points container
 * @param point Local point global coordinates structure (output)
 *
 * @pre IndexType must satisfy is_index_type constraint
 * @pre ContainerType and PointType must satisfy is_global_coordinates
 * constraint
 *
 * @code
 * // Example usage in host code
 * specfem::point::index<specfem::dimension::type::dim3> idx(ispec, iz, iy, ix);
 * specfem::point::global_coordinates<specfem::dimension::type::dim3>
 * point_coord; specfem::assembly::load_on_host(idx, mesh.points, point_coord);
 *
 * // Process loaded coordinate data
 * type_real x_val = point_coord.x;
 * type_real y_val = point_coord.y;
 * type_real z_val = point_coord.z;
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            specfem::data_access::is_global_coordinates<ContainerType>::value &&
            specfem::data_access::is_global_coordinates<PointType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_host(const IndexType &index,
                                              const ContainerType &container,
                                              PointType &point) {
  static_assert(IndexType::dimension_tag == ContainerType::dimension_tag,
                "Index and container dimension tags must match");
  static_assert(IndexType::dimension_tag == PointType::dimension_tag,
                "Index and point dimension tags must match");

  impl_load<false>(index, container, point);
}

} // namespace specfem::assembly
