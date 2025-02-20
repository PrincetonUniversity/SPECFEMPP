#pragma once

#include "compute/element_types/element_types.hpp"
#include "compute/impl/value_containers.hpp"
#include "enumerations/medium.hpp"
#include "medium/material_kernels.hpp"
#include "mesh/mesh.hpp"
#include "point/coordinates.hpp"
#include "point/kernels.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
/**
 * @brief Misfit kernels (Frechet derivatives) for every quadrature point in the
 * finite element mesh
 *
 */
struct kernels
    : public impl::value_containers<specfem::medium::material_kernels> {
public:
  /**
   * @name Constructors
   *
   */

  ///@{
  /**
   * @brief Default constructor
   *
   */
  kernels() = default;

  /**
   * @brief Construct a new kernels object
   *
   * @param nspec Total number of spectral elements
   * @param ngllz Number of quadrature points in z dimension
   * @param ngllx Number of quadrature points in x dimension
   * @param mapping mesh to compute mapping
   * @param tags Tags for every element in spectral element mesh
   */
  kernels(const int nspec, const int ngllz, const int ngllx,
          const specfem::compute::element_types &element_types);
  ///@}

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    impl::value_containers<specfem::medium::material_kernels>::copy_to_host();
  }

  void copy_to_device() {
    impl::value_containers<specfem::medium::material_kernels>::copy_to_device();
  }
};

/**
 * @defgroup ComputeKernelsDataAccess
 */

/**
 * @brief Load misfit kernels for a given quadrature point on the device
 *
 * @ingroup ComputeKernelsDataAccess
 *
 * @tparam PointKernelType Point kernel type. Needs to be of @ref
 * specfem::point::kernels
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param kernels Misfit kernels container
 * @param point_kernels Kernels at a given quadrature point (output)
 */
template <typename PointKernelType, typename IndexType,
          typename std::enable_if<IndexType::using_simd ==
                                      PointKernelType::simd::using_simd,
                                  int>::type = 0>
KOKKOS_FUNCTION void load_on_device(const IndexType &index,
                                    const kernels &kernels,
                                    PointKernelType &point_kernels) {
  const int ispec = kernels.property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  kernels.get_container<MediumTag, PropertyTag>().load_device_kernels(
      l_index, point_kernels);

  return;
}

/**
 * @brief Load misfit kernels for a given quadrature point on the host
 *
 * @ingroup ComputeKernelsDataAccess
 *
 * @tparam PointKernelType Point kernel type. Needs to be of @ref
 * specfem::point::kernels
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param kernels Misfit kernels container
 * @param point_kernels Kernels at a given quadrature point (output)
 */
template <typename PointKernelType, typename IndexType,
          typename std::enable_if<IndexType::using_simd ==
                                      PointKernelType::simd::using_simd,
                                  int>::type = 0>
void load_on_host(const IndexType &index, const kernels &kernels,
                  PointKernelType &point_kernels) {
  const int ispec = kernels.h_property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  kernels.get_container<MediumTag, PropertyTag>().load_host_kernels(
      l_index, point_kernels);

  return;
}

/**
 * @brief Store misfit kernels for a given quadrature point on the host
 *
 * @ingroup ComputeKernelsDataAccess
 *
 * @tparam PointKernelType Point kernel type. Needs to be of @ref
 * specfem::point::kernels
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param point_kernels Kernels at a given quadrature point
 * @param kernels Misfit kernels container
 */
template <typename PointKernelType, typename IndexType,
          typename std::enable_if<IndexType::using_simd ==
                                      PointKernelType::simd::using_simd,
                                  int>::type = 0>
void store_on_host(const IndexType &index, const PointKernelType &point_kernels,
                   const kernels &kernels) {
  const int ispec = kernels.h_property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  kernels.get_container<MediumTag, PropertyTag>().update_kernels_on_host(
      l_index, point_kernels);

  return;
}

/**
 * @brief Store misfit kernels for a given quadrature point on the device
 *
 * @ingroup ComputeKernelsDataAccess
 *
 * @tparam PointKernelType Point kernel type. Needs to be of @ref
 * specfem::point::kernels
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param point_kernels Kernels at a given quadrature point
 * @param kernels Misfit kernels container
 */
template <typename PointKernelType, typename IndexType,
          typename std::enable_if<IndexType::using_simd ==
                                      PointKernelType::simd::using_simd,
                                  int>::type = 0>
KOKKOS_FUNCTION void store_on_device(const IndexType &index,
                                     const PointKernelType &point_kernels,
                                     const kernels &kernels) {
  const int ispec = kernels.property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  kernels.get_container<MediumTag, PropertyTag>().update_kernels_on_device(
      l_index, point_kernels);

  return;
}

/**
 * @brief Add misfit kernels for a given quadrature point to the existing
 * kernels on the device
 *
 * @ingroup ComputeKernelsDataAccess
 *
 * @tparam PointKernelType Point kernel type. Needs to be of @ref
 * specfem::point::kernels
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param point_kernels Kernels at a given quadrature point
 * @param kernels Misfit kernels container
 */
template <typename IndexType, typename PointKernelType,
          typename std::enable_if<IndexType::using_simd ==
                                      PointKernelType::simd::using_simd,
                                  int>::type = 0>
KOKKOS_FUNCTION void add_on_device(const IndexType &index,
                                   const PointKernelType &point_kernels,
                                   const kernels &kernels) {

  const int ispec = kernels.property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  kernels.get_container<MediumTag, PropertyTag>().add_kernels_on_device(
      l_index, point_kernels);

  return;
}

/**
 * @brief Add misfit kernels for a given quadrature point to the existing
 * kernels on the host
 *
 * @ingroup ComputeKernelsDataAccess
 *
 * @tparam PointKernelType Point kernel type. Needs to be of @ref
 * specfem::point::kernels
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param point_kernels Kernels at a given quadrature point
 * @param kernels Misfit kernels container
 */
template <typename IndexType, typename PointKernelType,
          typename std::enable_if<IndexType::using_simd ==
                                      PointKernelType::simd::using_simd,
                                  int>::type = 0>
void add_on_host(const IndexType &index, const PointKernelType &point_kernels,
                 const kernels &kernels) {
  const int ispec = kernels.h_property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  kernels.get_container<MediumTag, PropertyTag>().add_kernels_on_host(
      l_index, point_kernels);

  return;
}

} // namespace compute
} // namespace specfem
