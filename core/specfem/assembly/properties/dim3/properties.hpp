#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/impl/domain_properties.hpp"
#include "specfem/assembly/impl/value_containers.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace specfem::assembly {

/**
 * @brief 3D spectral element material properties data container
 *
 * This template specialization provides storage and management of material
 * properties for 3-dimensional spectral elements. It stores properties such
 * as density, elastic constants, and medium-specific parameters at quadrature
 * points throughout the 3D spectral element mesh.
 *
 * The class inherits from value_containers which provides the underlying
 * storage mechanism and data access patterns for different material types
 * (elastic, acoustic, poroelastic) and property variations (isotropic,
 * anisotropic, Cosserat). The 3D implementation handles the additional
 * y-dimension compared to the 2D case.
 *
 * Related classes:
 * - specfem::assembly::impl::domain_properties: Base storage container
 * - specfem::assembly::mesh: 3D mesh geometry and connectivity
 * - specfem::mesh::materials: Material assignment per 3D element
 *
 * @code
 * // Typical usage in 3D assembly construction
 * specfem::assembly::properties<specfem::element::dimension_tag::dim3> props(
 *     nspec, ngllz, nglly, ngllx, materials, element_types);
 *
 * // Access 3D material properties at quadrature points
 * auto container = props.get_container<medium_tag, property_tag>();
 * @endcode
 */
template <>
struct properties<specfem::element::dimension_tag::dim3>
    : public impl::value_containers<specfem::element::dimension_tag::dim3,
                                    impl::domain_properties> {

  /**
   * @brief Default constructor.
   *
   * Initializes an empty 3D properties container with no allocated storage.
   */
  properties() = default;

  /**
   * @brief Construct 3D properties container from mesh and material data.
   *
   * Initializes the 3D material properties container by extracting material
   * properties from the mesh materials and storing them at quadrature points
   * for all 3D spectral elements. The constructor handles the full 3D tensor
   * grid of quadrature points (x, y, z directions).
   *
   * @param nspec Number of spectral elements in the 3D mesh
   * @param ngllz Number of quadrature points in z-direction (vertical)
   * @param nglly Number of quadrature points in y-direction
   * @param ngllx Number of quadrature points in x-direction (horizontal)
   * @param materials Material properties database indexed by 3D element
   * @param element_types Element type information for each 3D spectral element
   */
  properties(
      const int nspec, const int ngllz, const int nglly, const int ngllx,
      const specfem::mesh::materials<dimension_tag> &materials,
      const specfem::assembly::element_types<dimension_tag> &element_types);

  /**
   * @brief Copy 3D material properties data from device to host memory.
   *
   * Transfers all 3D material property data stored in device memory back to
   * host-accessible memory for post-processing, I/O operations, or debugging.
   * Handles the full 3D tensor storage efficiently.
   */
  void copy_to_host() {
    impl::value_containers<
        dimension_tag,
        specfem::assembly::impl::domain_properties>::copy_to_host();
  }

  /**
   * @brief Copy 3D material properties data from host to device memory.
   *
   * Transfers 3D material property data from host memory to device-accessible
   * memory for use in GPU-accelerated 3D computations. Optimized for the
   * 3D tensor structure of quadrature points.
   */
  void copy_to_device() {
    impl::value_containers<
        dimension_tag,
        specfem::assembly::impl::domain_properties>::copy_to_device();
  }
};
} // namespace specfem::assembly
