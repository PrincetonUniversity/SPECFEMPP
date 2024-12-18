#pragma once

#include "enumerations/medium.hpp"
#include "mesh/materials/materials.hpp"
#include "point/coordinates.hpp"
#include "compute/compute_mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
/**
 * @brief Misfit kernels (Frechet derivatives) for every quadrature point in the
 * finite element mesh
 *
 */
struct element_info{
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

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  element_info() = default;

  /**
   * @brief Construct a new kernels object
   *
   * @param nspec Total number of spectral elements
   * @param ngllz Number of quadrature points in z dimension
   * @param ngllx Number of quadrature points in x dimension
   * @param mapping mesh to compute mapping
   * @param tags Tags for every element in spectral element mesh
   */
  element_info(const int nspec, const int ngllz, const int ngllx,
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
  }

  void copy_to_device() {
    Kokkos::deep_copy(element_types, h_element_types);
    Kokkos::deep_copy(element_property, h_element_property);
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
};

} // namespace compute
} // namespace specfem

