#pragma once

#include "compute/compute_mesh.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh.hpp"
#include "point/coordinates.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
/**
 * @brief Element types for every quadrature point in the
 * finite element mesh
 *
 */
struct element_types {
protected:
  using MediumTagViewType =
      Kokkos::View<specfem::element::medium_tag *,
                   Kokkos::DefaultHostExecutionSpace>; ///< Underlying view type
                                                       ///< to store medium tags
  using PropertyTagViewType =
      Kokkos::View<specfem::element::property_tag *,
                   Kokkos::DefaultHostExecutionSpace>; ///< Underlying view type
                                                       ///< to store property
                                                       ///< tags
  using BoundaryViewType =
      Kokkos::View<specfem::element::boundary_tag *,
                   Kokkos::DefaultHostExecutionSpace>; ///< Underlying view type
                                                       ///< to store
                                                       ///< boundary tags

  using IndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices of
                                                          ///< elements

public:
  int nspec; ///< total number of spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension

  MediumTagViewType medium_tags;     ///< View to store medium tags
  PropertyTagViewType property_tags; ///< View to store property tags
  BoundaryViewType boundary_tags;    ///< View to store boundary tags

  /**
   * @brief Default constructor
   *
   */
  element_types() = default;

  /**
   * @brief Construct a new properties object from mesh information
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of quadrature points in z direction
   * @param ngllx Number of quadrature points in x direction
   * @param mapping Mapping of spectral element index from mesh to assembly
   * @param tags Element Tags for every spectral element
   */
  element_types(
      const int nspec, const int ngllz, const int ngllx,
      const specfem::compute::mesh_to_compute_mapping &mapping,
      const specfem::mesh::tags<specfem::dimension::type::dim2> &tags);

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag) const;

  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag) const;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag,
                       const specfem::element::property_tag property) const;

  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag,
                         const specfem::element::property_tag property) const;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag,
                       const specfem::element::property_tag property,
                       const specfem::element::boundary_tag boundary) const;

  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag,
                         const specfem::element::property_tag property,
                         const specfem::element::boundary_tag boundary) const;

  specfem::element::medium_tag get_medium_tag(const int ispec) const {
    return medium_tags(ispec);
  }

  specfem::element::property_tag get_property_tag(const int ispec) const {
    return property_tags(ispec);
  }

  specfem::element::boundary_tag get_boundary_tag(const int ispec) const {
    return boundary_tags(ispec);
  }

private:
#define MEDIUM_TAG_VARIABLES(DIMENSION_TAG, MEDIUM_TAG)                        \
  IndexViewType CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),        \
                                     GET_NAME(MEDIUM_TAG));                    \
  IndexViewType::HostMirror CREATE_VARIABLE_NAME(                              \
      h_elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG));

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(MEDIUM_TAG_VARIABLES,
                                 WHERE(DIMENSION_TAG_DIM2)
                                     WHERE(MEDIUM_TAG_ELASTIC,
                                           MEDIUM_TAG_ACOUSTIC))

#undef MEDIUM_TAG_VARIABLES

#define MATERIAL_SYSTEMS_VARIABLE_NAMES(DIMENSION_TAG, MEDIUM_TAG,             \
                                        PROPERTY_TAG)                          \
  IndexViewType CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),        \
                                     GET_NAME(MEDIUM_TAG),                     \
                                     GET_NAME(PROPERTY_TAG));                  \
  IndexViewType::HostMirror CREATE_VARIABLE_NAME(                              \
      h_elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),               \
      GET_NAME(PROPERTY_TAG));

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      MATERIAL_SYSTEMS_VARIABLE_NAMES,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef MATERIAL_SYSTEMS_VARIABLE_NAMES

#define ELEMENT_TYPES_VARIABLE_NAMES(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG,  \
                                     BOUNDARY_TAG)                             \
  IndexViewType CREATE_VARIABLE_NAME(                                          \
      elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),                 \
      GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG));                         \
  IndexViewType::HostMirror CREATE_VARIABLE_NAME(                              \
      h_elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),               \
      GET_NAME(PROPERTY_TAG), GET_NAME(BOUNDARY_TAG));

  CALL_MACRO_FOR_ALL_ELEMENT_TYPES(
      ELEMENT_TYPES_VARIABLE_NAMES,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)
              WHERE(BOUNDARY_TAG_NONE, BOUNDARY_TAG_ACOUSTIC_FREE_SURFACE,
                    BOUNDARY_TAG_STACEY,
                    BOUNDARY_TAG_COMPOSITE_STACEY_DIRICHLET))

#undef ELEMENT_TYPES_VARIABLE_NAMES
};

} // namespace compute
} // namespace specfem
