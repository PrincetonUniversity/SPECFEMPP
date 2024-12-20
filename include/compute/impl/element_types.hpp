#pragma once

#include "compute/compute_mesh.hpp"
#include "enumerations/medium.hpp"
#include "mesh/materials/materials.hpp"
#include "point/coordinates.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {

namespace impl {
/**
 * @brief Element types for every quadrature point in the
 * finite element mesh
 *
 */
struct element_types {
protected:
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using MediumTagViewType =
      Kokkos::View<specfem::element::medium_tag *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store medium tags
  using PropertyTagViewType =
      Kokkos::View<specfem::element::property_tag *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store property tags
  using BoundaryViewType =
      Kokkos::View<specfem::element::boundary_tag *,
                   Kokkos::HostSpace>; //< Underlying view type to store
                                       // boundary tags

public:
  int nspec;                     ///< total number of spectral elements
  int ngllz;                     ///< number of quadrature points in z dimension
  int ngllx;                     ///< number of quadrature points in x dimension
  MediumTagViewType medium_tags; ///< Medium tag for every spectral element
  PropertyTagViewType property_tags; ///< Property tag for every spectral
                                     ///< element
  BoundaryViewType boundary_tags;    ///< Boundary tags for every element in the
                                     ///< mesh
  MediumTagViewType::HostMirror h_medium_tags;     ///< Host mirror of @ref
                                                   ///< medium_tags
  PropertyTagViewType::HostMirror h_property_tags; ///< Host mirror of @ref
                                                   ///< property_tags

  IndexViewType property_index_mapping;
  IndexViewType::HostMirror h_property_index_mapping;

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

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    Kokkos::deep_copy(h_medium_tags, medium_tags);
    Kokkos::deep_copy(h_property_tags, property_tags);
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
  }

  void copy_to_device() {
    Kokkos::deep_copy(medium_tags, h_medium_tags);
    Kokkos::deep_copy(property_tags, h_property_tags);
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
  }

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

  /**
   * @brief Get the indices of elements of a given type as a view on the host
   *
   * @param medium Medium tag of the elements
   * @param boundary Boundary tag of the elements
   * @return Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> View of
   * the indices of elements of the given type
   */
  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
  get_elements_on_host(const specfem::element::medium_tag medium,
                       const specfem::element::boundary_tag boundary) const;
};

} // namespace impl
} // namespace compute
} // namespace specfem
