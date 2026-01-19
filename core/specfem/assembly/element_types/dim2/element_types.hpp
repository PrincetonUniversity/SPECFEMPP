#pragma once

#include "enumerations/medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief 2D spectral element type classification and indexing container
 *
 * This template specialization provides storage and management for element
 * type information in 2D spectral element meshes. It stores medium types
 * (elastic P-SV, elastic SH, acoustic, poroelastic), material properties
 * (isotropic, anisotropic, Cosserat), and boundary conditions for each
 * spectral element.
 *
 * The class enables efficient querying and filtering of elements by their
 * physical characteristics, which is essential for heterogeneous 2D media
 * where different wave equations apply to different regions. It maintains
 * both host and device views for optimal performance in hybrid CPU-GPU
 * computations.
 *
 * Related classes:
 * - specfem::assembly::mesh: 2D mesh geometry and connectivity
 * - specfem::mesh::tags: Element classification input data
 * - specfem::assembly::properties: Material properties per element
 *
 * @code
 * // Construct 2D element types from mesh data
 * specfem::assembly::element_types<specfem::dimension::type::dim2> etypes(
 *     nspec, ngllz, ngllx, mesh, tags);
 *
 * // Query elastic P-SV elements for solver
 * auto psv_elements = etypes.get_elements_on_device(
 *     specfem::element::medium_tag::elastic_psv);
 *
 * // Get isotropic elastic elements
 * auto iso_elements = etypes.get_elements_on_host(
 *     specfem::element::medium_tag::elastic_psv,
 *     specfem::element::property_tag::isotropic);
 * @endcode
 */
template <> struct element_types<specfem::dimension::type::dim2> {
protected:
  /**
   * @brief Kokkos view type for storing medium tags in host memory.
   *
   * Stores the medium type (elastic_psv, elastic_sh, acoustic, poroelastic,
   * etc.) for each spectral element. Host execution space enables CPU-side
   * operations for initialization and debugging.
   */
  using MediumTagViewType = Kokkos::View<specfem::element::medium_tag *,
                                         Kokkos::DefaultHostExecutionSpace>;

  /**
   * @brief Kokkos view type for storing property tags in host memory.
   *
   * Stores the material property type (isotropic, anisotropic,
   * isotropic_cosserat) for each spectral element. Host execution space allows
   * for efficient host-side filtering and element classification operations.
   */
  using PropertyTagViewType = Kokkos::View<specfem::element::property_tag *,
                                           Kokkos::DefaultHostExecutionSpace>;

  /**
   * @brief Kokkos view type for storing boundary condition tags in host memory.
   *
   * Stores the boundary condition type (none, acoustic_free_surface, stacey,
   * composite_stacey_dirichlet) for each spectral element. Host storage enables
   * efficient boundary condition setup and validation.
   */
  using BoundaryViewType = Kokkos::View<specfem::element::boundary_tag *,
                                        Kokkos::DefaultHostExecutionSpace>;

  /**
   * @brief Kokkos view type for storing element indices in device memory.
   *
   * Stores integer indices of spectral elements for device-side computations.
   * Default execution space enables optimal performance for GPU kernels and
   * parallel element processing operations.
   */
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

public:
  int nspec; ///< total number of spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension

  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag

  MediumTagViewType medium_tags;     ///< View to store medium tags
  PropertyTagViewType property_tags; ///< View to store property tags
  BoundaryViewType boundary_tags;    ///< View to store boundary tags

  /**
   * @brief Default constructor.
   *
   * Initializes an empty 2D element types container with no allocated storage.
   */
  element_types() = default;

  /**
   * @brief Construct 2D element types container from mesh and tag data.
   *
   * Initializes the element type classification system by extracting medium,
   * property, and boundary tags from the mesh tag data. Creates indexed views
   * for efficient element filtering and builds both host and device storage
   * for each combination of element characteristics.
   *
   * @param nspec Number of spectral elements in the 2D mesh
   * @param ngllz Number of quadrature points in z-direction (vertical)
   * @param ngllx Number of quadrature points in x-direction (horizontal)
   * @param mesh 2D assembly mesh containing geometry and connectivity
   * @param tags Element classification data containing medium, property, and
   * boundary tags
   *
   * @code
   * specfem::assembly::element_types<specfem::dimension::type::dim2> etypes(
   *     1000, 5, 5, assembly_mesh, mesh_tags);
   * @endcode
   */
  element_types(const int nspec, const int ngllz, const int ngllx,
                const specfem::assembly::mesh<dimension_tag> &mesh,
                const specfem::mesh::tags<dimension_tag> &tags);

  /**
   * @brief Get elements with specified medium type in host memory.
   *
   * @param tag Medium type (elastic_psv, elastic_sh, acoustic, etc.)
   * @return Kokkos view containing element indices for host access
   */
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag) const;

  /**
   * @brief Get count of elements with specified medium type.
   *
   * @param tag Medium type to count
   * @return Number of elements with the specified medium type
   */
  int get_number_of_elements(const specfem::element::medium_tag tag) const {
    return get_elements_on_host(tag).extent(0);
  }

  /**
   * @brief Get elements with specified medium type in device memory.
   *
   * @param tag Medium type (elastic_psv, elastic_sh, acoustic, etc.)
   * @return Kokkos view containing element indices for device access
   */
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag) const;

  /**
   * @brief Get elements with specified medium and property types in host
   * memory.
   *
   * @param tag Medium type (elastic_psv, elastic_sh, acoustic, etc.)
   * @param property Property type (isotropic, anisotropic, isotropic_cosserat)
   * @return Kokkos view containing element indices for host access
   */
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag,
                       const specfem::element::property_tag property) const;

  /**
   * @brief Get count of elements with specified medium and property types.
   *
   * @param tag Medium type to count
   * @param property Property type to count
   * @return Number of elements matching both medium and property types
   */
  int get_number_of_elements(
      const specfem::element::medium_tag tag,
      const specfem::element::property_tag property) const {
    return get_elements_on_host(tag, property).extent(0);
  }

  /**
   * @brief Get elements with specified medium and property types in device
   * memory.
   *
   * @param tag Medium type (elastic_psv, elastic_sh, acoustic, etc.)
   * @param property Property type (isotropic, anisotropic, isotropic_cosserat)
   * @return Kokkos view containing element indices for device access
   */
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag,
                         const specfem::element::property_tag property) const;

  /**
   * @brief Get elements with specified medium, property, and boundary types in
   * host memory.
   *
   * @param tag Medium type (elastic_psv, elastic_sh, acoustic, etc.)
   * @param property Property type (isotropic, anisotropic, isotropic_cosserat)
   * @param boundary Boundary condition type (none, acoustic_free_surface,
   * stacey, etc.)
   * @return Kokkos view containing element indices for host access
   */
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag,
                       const specfem::element::property_tag property,
                       const specfem::element::boundary_tag boundary) const;

  /**
   * @brief Get count of elements with specified medium, property, and boundary
   * types.
   *
   * @param tag Medium type to count
   * @param property Property type to count
   * @param boundary Boundary condition type to count
   * @return Number of elements matching all three classification criteria
   */
  int get_number_of_elements(
      const specfem::element::medium_tag tag,
      const specfem::element::property_tag property,
      const specfem::element::boundary_tag boundary) const {
    return get_elements_on_host(tag, property, boundary).extent(0);
  }

  /**
   * @brief Get elements with specified medium, property, and boundary types in
   * device memory.
   *
   * @param tag Medium type (elastic_psv, elastic_sh, acoustic, etc.)
   * @param property Property type (isotropic, anisotropic, isotropic_cosserat)
   * @param boundary Boundary condition type (none, acoustic_free_surface,
   * stacey, etc.)
   * @return Kokkos view containing element indices for device access
   */
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag,
                         const specfem::element::property_tag property,
                         const specfem::element::boundary_tag boundary) const;

  /**
   * @brief Get medium type for a specific spectral element.
   *
   * @param ispec Spectral element index
   * @return Medium type (elastic_psv, elastic_sh, acoustic, poroelastic, etc.)
   */
  specfem::element::medium_tag get_medium_tag(const int ispec) const {
    return medium_tags(ispec);
  }

  /**
   * @brief Get property type for a specific spectral element.
   *
   * @param ispec Spectral element index
   * @return Property type (isotropic, anisotropic, isotropic_cosserat)
   */
  specfem::element::property_tag get_property_tag(const int ispec) const {
    return property_tags(ispec);
  }

  /**
   * @brief Get boundary condition type for a specific spectral element.
   *
   * @param ispec Spectral element index
   * @return Boundary type (none, acoustic_free_surface, stacey,
   * composite_stacey_dirichlet)
   */
  specfem::element::boundary_tag get_boundary_tag(const int ispec) const {
    return boundary_tags(ispec);
  }

private:
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T)),
                      DECLARE((IndexViewType, elements),
                              (IndexViewType::HostMirror, h_elements)))

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T),
                       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC,
                                    ISOTROPIC_COSSERAT)),
                      DECLARE((IndexViewType, elements),
                              (IndexViewType::HostMirror, h_elements)))

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T),
                       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
                       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      DECLARE((IndexViewType, elements),
                              (IndexViewType::HostMirror, h_elements)))
};

} // namespace specfem::assembly
