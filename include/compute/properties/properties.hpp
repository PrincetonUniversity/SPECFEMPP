#ifndef _COMPUTE_PROPERTIES_HPP
#define _COMPUTE_PROPERTIES_HPP

#include "enumerations/specfem_enums.hpp"
#include "impl/material_properties.hpp"
#include "impl/properties_container.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "material/material.hpp"
#include "point/coordinates.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace specfem {
namespace compute {

/**
 * @brief Material properties at every quadrature point in the finite element
 * mesh
 *
 */
struct properties {
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
  IndexViewType property_index_mapping;
  IndexViewType::HostMirror h_property_index_mapping;
  MediumTagViewType element_types;      ///< Medium Tag for every spectral
                                        ///< element
  PropertyTagViewType element_property; ///< Property Tag for every spectral
                                        ///< element
  MediumTagViewType::HostMirror h_element_types;      ///< Host mirror of
                                                      ///< @ref element_types
  PropertyTagViewType::HostMirror h_element_property; ///< Host mirror of
                                                      ///< @ref element_property

  specfem::compute::impl::properties::material_property<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>
      elastic_isotropic; ///< Elastic isotropic material properties
  specfem::compute::impl::properties::material_property<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic>
      elastic_anisotropic; ///< Elastic anisotropic material properties
  specfem::compute::impl::properties::material_property<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>
      acoustic_isotropic; ///< Acoustic isotropic material properties

  /**
   * @name Constructors
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  properties() = default;

  /**
   * @brief Construct a new properties object from mesh information
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of quadrature points in z direction
   * @param ngllx Number of quadrature points in x direction
   * @param mapping Mapping of spectral element index from mesh to assembly
   * @param tags Element Tags for every spectral element
   * @param materials Material properties for every spectral element
   */
  properties(const int nspec, const int ngllz, const int ngllx,
             const specfem::compute::mesh_to_compute_mapping &mapping,
             const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
             const specfem::mesh::materials &materials);

  ///@}

  /**
   * @brief Get the indices of elements of a given type as a view on the device
   *
   * @param medium Medium tag of the elements
   * @return Kokkos::View<int *, Kokkos::LayoutLeft,
   * Kokkos::DefaultExecutionSpace> View of the indices of elements of the given
   * type
   */
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag medium) const;

  /**
   * @brief Get the indices of elements of a given type as a view on the device
   *
   * @param medium Medium tag of the elements
   * @param property Property tag of the elements
   * @return Kokkos::View<int *, Kokkos::LayoutLeft,
   * Kokkos::DefaultExecutionSpace> View of the indices of elements of the given
   * type
   */
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag medium,
                         const specfem::element::property_tag property) const;

  /**
   * @brief Get the indices of elements of a given type as a view on the host
   *
   * @param medium Medium tag of the elements
   * @return Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> View of
   * the indices of elements of the given type
   */
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
  get_elements_on_host(const specfem::element::medium_tag medium) const;

  /**
   * @brief Get the indices of elements of a given type as a view on the host
   *
   * @param medium Medium tag of the elements
   * @param property Property tag of the elements
   * @return Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> View of
   * the indices of elements of the given type
   */
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
  get_elements_on_host(const specfem::element::medium_tag medium,
                       const specfem::element::property_tag property) const;
};

/**
 * @defgroup ComputePropertiesDataAccess
 */

/**
 * @brief Load the material properties at a given quadrature point on the device
 *
 * @ingroup ComputePropertiesDataAccess
 *
 * @tparam PointPropertiesType Point properties type. Needs to be of @ref
 * specfem::point::properties
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param lcoord Index of the quadrature point
 * @param properties Material properties container
 * @param point_properties Material properties at the given quadrature point
 * (output)
 */
template <typename PointPropertiesType, typename IndexType,
          typename std::enable_if_t<IndexType::using_simd ==
                                        PointPropertiesType::simd::using_simd,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const IndexType &lcoord,
               const specfem::compute::properties &properties,
               PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionType = PointPropertiesType::dimension;

  static_assert(DimensionType == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    properties.elastic_isotropic.load_device_properties(l_index,
                                                        point_properties);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::anisotropic)) {
    properties.elastic_anisotropic.load_device_properties(l_index,
                                                          point_properties);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    properties.acoustic_isotropic.load_device_properties(l_index,
                                                         point_properties);
  } else {
    static_assert("Material type not implemented");
  }
}

/**
 * @brief Store the material properties at a given quadrature point on the
 * device
 *
 * @ingroup ComputePropertiesDataAccess
 *
 * @tparam PointPropertiesType Point properties type. Needs to be of @ref
 * specfem::point::properties
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param lcoord Index of the quadrature point
 * @param properties Material properties container
 * @param point_properties Material properties at the given quadrature point
 */
template <typename PointPropertiesType, typename IndexType,
          typename std::enable_if_t<IndexType::using_simd ==
                                        PointPropertiesType::simd::using_simd,
                                    int> = 0>
void load_on_host(const IndexType &lcoord,
                  const specfem::compute::properties &properties,
                  PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.h_property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionType = PointPropertiesType::dimension;

  static_assert(DimensionType == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    properties.elastic_isotropic.load_host_properties(l_index,
                                                      point_properties);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::anisotropic)) {
    properties.elastic_anisotropic.load_host_properties(l_index,
                                                        point_properties);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    properties.acoustic_isotropic.load_host_properties(l_index,
                                                       point_properties);
  } else {
    static_assert("Material type not implemented");
  }
}

/**
 * @brief Store the material properties at a given quadrature point on the host
 *
 * @ingroup ComputePropertiesDataAccess
 *
 * @tparam PointPropertiesType Point properties type. Needs to be of @ref
 * specfem::point::properties
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param lcoord Index of the quadrature point
 * @param properties Material properties container
 * @param point_properties Material properties at the given quadrature point
 */
template <typename PointPropertiesType, typename IndexType,
          typename std::enable_if_t<IndexType::using_simd ==
                                        PointPropertiesType::simd::using_simd,
                                    int> = 0>
void store_on_host(const IndexType &lcoord,
                   const specfem::compute::properties &properties,
                   const PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  const int index = properties.h_property_index_mapping(ispec);

  IndexType l_index = lcoord;

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionType = PointPropertiesType::dimension;

  static_assert(DimensionType == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    properties.elastic_isotropic.assign(l_index, point_properties);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::anisotropic)) {
    properties.elastic_anisotropic.assign(l_index, point_properties);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    properties.acoustic_isotropic.assign(l_index, point_properties);
  } else {
    static_assert("Material type not implemented");
  }
}
} // namespace compute
} // namespace specfem

#endif
