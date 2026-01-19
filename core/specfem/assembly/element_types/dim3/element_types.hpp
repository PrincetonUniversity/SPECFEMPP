#pragma once

#include "enumerations/medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief 3D spectral element type classification and indexing container
 *
 * This template specialization provides storage and management for element
 * type information in 3D spectral element meshes. It stores medium types
 * (primarily elastic for 3D applications), material properties (isotropic,
 * anisotropic, Cosserat), and boundary conditions for each 3D spectral element.
 *
 * The class enables efficient querying and filtering of 3D elements by their
 * physical characteristics, which is essential for heterogeneous 3D media
 * in applications such as seismic wave propagation, structural analysis,
 * and geophysical modeling. It maintains both host and device views for
 * optimal performance in hybrid CPU-GPU 3D computations.
 *
 * Supported 3D medium types:
 * - ELASTIC: 3D elastic wave propagation (primary focus for 3D)
 *
 * The 3D implementation is currently optimized for elastic media with
 * support for various material property combinations and boundary conditions.
 *
 * Related classes:
 * - specfem::assembly::mesh: 3D mesh geometry and connectivity
 * - specfem::mesh::tags: Element classification input data
 * - specfem::assembly::properties: Material properties per 3D element
 *
 * @code
 * // Construct 3D element types from mesh data
 * specfem::assembly::element_types<specfem::dimension::type::dim3> etypes(
 *     nspec, ngllz, nglly, ngllx, mesh, tags);
 *
 * // Query 3D elastic elements for solver
 * auto elastic_elements = etypes.get_elements_on_device(
 *     specfem::element::medium_tag::elastic);
 *
 * // Get isotropic elastic elements in 3D
 * auto iso_elements = etypes.get_elements_on_host(
 *     specfem::element::medium_tag::elastic,
 *     specfem::element::property_tag::isotropic);
 * @endcode
 */
template <> struct element_types<specfem::dimension::type::dim3> {
protected:
  /**
   * @brief Kokkos view type for storing 3D medium tags in host memory.
   *
   * Stores the medium type (primarily elastic for 3D applications) for each
   * 3D spectral element. Host execution space enables CPU-side operations
   * for 3D mesh initialization and debugging.
   */
  using MediumTagViewType = Kokkos::View<specfem::element::medium_tag *,
                                         Kokkos::DefaultHostExecutionSpace>;

  /**
   * @brief Kokkos view type for storing 3D property tags in host memory.
   *
   * Stores the material property type (isotropic, anisotropic,
   * isotropic_cosserat) for each 3D spectral element. Host execution space
   * allows for efficient host-side filtering and 3D element classification
   * operations.
   */
  using PropertyTagViewType = Kokkos::View<specfem::element::property_tag *,
                                           Kokkos::DefaultHostExecutionSpace>;

  /**
   * @brief Kokkos view type for storing 3D boundary condition tags in host
   * memory.
   *
   * Stores the boundary condition type for each 3D spectral element. Host
   * storage enables efficient 3D boundary condition setup and validation
   * for complex 3D geometries.
   */
  using BoundaryViewType = Kokkos::View<specfem::element::boundary_tag *,
                                        Kokkos::DefaultHostExecutionSpace>;

  /**
   * @brief Kokkos view type for storing 3D element indices in device memory.
   *
   * Stores integer indices of 3D spectral elements for device-side
   * computations. Default execution space enables optimal performance for 3D
   * GPU kernels and parallel 3D element processing operations.
   */
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

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
   * @brief Default constructor.
   *
   * Initializes an empty 3D element types container with no allocated storage.
   */
  element_types() = default;

  /**
   * @brief Constructor for element_types class in 3D assembly
   *
   * Initializes the element types container with spectral element mesh
   * parameters and mesh configuration data for 3D finite element assembly
   * operations.
   *
   * @param nspec Total number of spectral elements in the mesh
   * @param ngllz Number of Gauss-Lobatto-Legendre points in the z-direction
   * @param nglly Number of Gauss-Lobatto-Legendre points in the y-direction
   * @param ngllx Number of Gauss-Lobatto-Legendre points in the x-direction
   * @param mesh Reference to the 3D assembly mesh containing geometric and
   * topological information
   * @param tags Reference to the mesh tags containing element type
   * classifications and material properties
   */
  element_types(const int nspec, const int ngllz, const int nglly,
                const int ngllx,
                const specfem::assembly::mesh<dimension_tag> &mesh,
                const specfem::mesh::tags<dimension_tag> &tags);

  /**
   * @brief Get 3D elements with specified medium type in host memory.
   *
   * @param tag Medium type (primarily elastic for 3D applications)
   * @return Kokkos view containing 3D element indices for host access
   */
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag) const;

  /**
   * @brief Get count of 3D elements with specified medium type.
   *
   * @param tag Medium type to count in 3D mesh
   * @return Number of 3D elements with the specified medium type
   */
  int get_number_of_elements(const specfem::element::medium_tag tag) const {
    return get_elements_on_host(tag).extent(0);
  }

  /**
   * @brief Get 3D elements with specified medium type in device memory.
   *
   * @param tag Medium type (primarily elastic for 3D applications)
   * @return Kokkos view containing 3D element indices for device access
   */
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag) const;

  /**
   * @brief Get 3D elements with specified medium and property types in host
   * memory.
   *
   * @param tag Medium type (primarily elastic for 3D applications)
   * @param property Property type (isotropic, anisotropic, isotropic_cosserat)
   * @return Kokkos view containing 3D element indices for host access
   */
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag,
                       const specfem::element::property_tag property) const;

  /**
   * @brief Get count of 3D elements with specified medium and property types.
   *
   * @param tag Medium type to count in 3D mesh
   * @param property Property type to count in 3D mesh
   * @return Number of 3D elements matching both medium and property types
   */
  int get_number_of_elements(
      const specfem::element::medium_tag tag,
      const specfem::element::property_tag property) const {
    return get_elements_on_host(tag, property).extent(0);
  }

  /**
   * @brief Get 3D elements with specified medium and property types in device
   * memory.
   *
   * @param tag Medium type (primarily elastic for 3D applications)
   * @param property Property type (isotropic, anisotropic, isotropic_cosserat)
   * @return Kokkos view containing 3D element indices for device access
   */
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag,
                         const specfem::element::property_tag property) const;

  /**
   * @brief Get 3D elements with specified medium, property, and boundary types
   * in host memory.
   *
   * @param tag Medium type (primarily elastic for 3D applications)
   * @param property Property type (isotropic, anisotropic, isotropic_cosserat)
   * @param boundary Boundary condition type for 3D elements
   * @return Kokkos view containing 3D element indices for host access
   */
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
  get_elements_on_host(const specfem::element::medium_tag tag,
                       const specfem::element::property_tag property,
                       const specfem::element::boundary_tag boundary) const;

  /**
   * @brief Get count of 3D elements with specified medium, property, and
   * boundary types.
   *
   * @param tag Medium type to count in 3D mesh
   * @param property Property type to count in 3D mesh
   * @param boundary Boundary condition type to count in 3D mesh
   * @return Number of 3D elements matching all three classification criteria
   */
  int get_number_of_elements(
      const specfem::element::medium_tag tag,
      const specfem::element::property_tag property,
      const specfem::element::boundary_tag boundary) const {
    return get_elements_on_host(tag, property, boundary).extent(0);
  }

  /**
   * @brief Get 3D elements with specified medium, property, and boundary types
   * in device memory.
   *
   * @param tag Medium type (primarily elastic for 3D applications)
   * @param property Property type (isotropic, anisotropic, isotropic_cosserat)
   * @param boundary Boundary condition type for 3D elements
   * @return Kokkos view containing 3D element indices for device access
   */
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
  get_elements_on_device(const specfem::element::medium_tag tag,
                         const specfem::element::property_tag property,
                         const specfem::element::boundary_tag boundary) const;

  /**
   * @brief Get medium type for a specific 3D spectral element.
   *
   * @param ispec 3D spectral element index
   * @return Medium type (primarily elastic for 3D applications)
   */
  specfem::element::medium_tag get_medium_tag(const int ispec) const {
    return medium_tags(ispec);
  }

  /**
   * @brief Get property type for a specific 3D spectral element.
   *
   * @param ispec 3D spectral element index
   * @return Property type (isotropic, anisotropic, isotropic_cosserat)
   */
  specfem::element::property_tag get_property_tag(const int ispec) const {
    return property_tags(ispec);
  }

  /**
   * @brief Get boundary condition type for a specific 3D spectral element.
   *
   * @param ispec 3D spectral element index
   * @return Boundary type for the 3D element
   */
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
