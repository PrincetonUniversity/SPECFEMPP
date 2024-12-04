#pragma once

#include "enumerations/medium.hpp"
#include "impl/material_kernels.hpp"
#include "mesh/materials/materials.hpp"
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
struct kernels {
private:
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using MediumTagViewType =
      Kokkos::View<specfem::element::medium_tag *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store medium tags
  using PropertyTagViewType =
      Kokkos::View<specfem::element::property_tag *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store property tags

public:
  int nspec; ///< total number of spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension
  MediumTagViewType element_types; ///< Medium tag for every spectral element
  PropertyTagViewType element_property; ///< Property tag for every spectral
                                        ///< element
  MediumTagViewType::HostMirror h_element_types;      ///< Host mirror of @ref
                                                      ///< element_types
  PropertyTagViewType::HostMirror h_element_property; ///< Host mirror of @ref
                                                      ///< element_property

  IndexViewType property_index_mapping;
  IndexViewType::HostMirror h_property_index_mapping;

  specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>
      elastic_isotropic; ///< Elastic isotropic material kernels

  specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>
      acoustic_isotropic; ///< Acoustic isotropic material kernels

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
          const specfem::compute::mesh_to_compute_mapping &mapping,
          const specfem::mesh::tags<specfem::dimension::type::dim2> &tags);
  ///@}

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    Kokkos::deep_copy(h_element_types, element_types);
    Kokkos::deep_copy(h_element_property, element_property);
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    elastic_isotropic.copy_to_host();
    acoustic_isotropic.copy_to_host();
  }

  void copy_to_device() {
    Kokkos::deep_copy(element_types, h_element_types);
    Kokkos::deep_copy(element_property, h_element_property);
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
    elastic_isotropic.copy_to_device();
    acoustic_isotropic.copy_to_device();
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

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.load_device_kernels(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.load_device_kernels(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

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

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.load_host_kernels(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.load_host_kernels(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

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

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.update_kernels_on_host(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.update_kernels_on_host(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

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

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.update_kernels_on_device(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.update_kernels_on_device(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

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

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.add_kernels_on_device(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.add_kernels_on_device(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

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

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.add_kernels_on_host(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.add_kernels_on_host(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

  return;
}

} // namespace compute
} // namespace specfem
