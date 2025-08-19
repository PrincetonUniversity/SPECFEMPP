#pragma once

#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

template <> struct element_types<specfem::dimension::type::dim3> {
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
  int nglly; ///< number of quadrature points in y dimension
  int ngllx; ///< number of quadrature points in x dimension

  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag

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
   * @param nglly Number of quadrature points in y direction
   * @param ngllx Number of quadrature points in x direction
   * @param mesh Mesh information
   * @param tags Element Tags for every spectral element
   */
  element_types(const int nspec, const int ngllz, const int nglly,
                const int ngllx,
                const specfem::assembly::mesh<dimension_tag> &mesh,
                const specfem::mesh::tags<dimension_tag> &tags);

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag) const;

  int get_number_of_elements(const specfem::element::medium_tag tag) const {
    return get_elements_on_host(tag).extent(0);
  }

  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag) const;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag,
                       const specfem::element::property_tag property) const;

  int get_number_of_elements(
      const specfem::element::medium_tag tag,
      const specfem::element::property_tag property) const {
    return get_elements_on_host(tag, property).extent(0);
  }

  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag,
                         const specfem::element::property_tag property) const;

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag,
                       const specfem::element::property_tag property,
                       const specfem::element::boundary_tag boundary) const;

  int get_number_of_elements(
      const specfem::element::medium_tag tag,
      const specfem::element::property_tag property,
      const specfem::element::boundary_tag boundary) const {
    return get_elements_on_host(tag, property, boundary).extent(0);
  }

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
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                      DECLARE((IndexViewType, elements),
                              (IndexViewType::HostMirror, h_elements)))

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC),
                       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC,
                                    ISOTROPIC_COSSERAT)),
                      DECLARE((IndexViewType, elements),
                              (IndexViewType::HostMirror, h_elements)))

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC),
                       PROPERTY_TAG(ISOTROPIC), BOUNDARY_TAG(NONE)),
                      DECLARE((IndexViewType, elements),
                              (IndexViewType::HostMirror, h_elements)))
};

} // namespace specfem::assembly
