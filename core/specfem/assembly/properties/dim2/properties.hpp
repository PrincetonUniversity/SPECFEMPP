#pragma once

#include "enumerations/interface.hpp"

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
 * @brief 2D spectral element material properties data container
 *
 * This template specialization provides storage and management of material
 * properties for 2-dimensional spectral elements. It stores properties such
 * as density, elastic constants, and medium-specific parameters at quadrature
 * points throughout the spectral element mesh.
 *
 * The class inherits from value_containers which provides the underlying
 * storage mechanism and data access patterns for different material types
 * (elastic, acoustic, poroelastic) and property variations (isotropic,
 * anisotropic, Cosserat).
 *
 * Related classes:
 * - specfem::assembly::impl::domain_properties: Base storage container
 * - specfem::assembly::mesh: Mesh geometry and connectivity
 * - specfem::mesh::materials: Material assignment per element
 */
template <>
struct properties<specfem::element::dimension_tag::dim2>
    : public impl::value_containers<specfem::element::dimension_tag::dim2,
                                    impl::domain_properties> {

  /**
   * @brief Default constructor.
   *
   * Initializes an empty properties container with no allocated storage.
   */
  properties() = default;

  /**
   * @brief Construct properties container from mesh and material data.
   *
   * Initializes the material properties container by extracting material
   * properties from the mesh materials and storing them at quadrature points
   * for all spectral elements.
   *
   * @param nspec Number of spectral elements in the mesh
   * @param ngllz Number of quadrature points in z-direction (vertical)
   * @param ngllx Number of quadrature points in x-direction (horizontal)
   * @param element_types Element type information for each spectral element
   * @param mesh Spectral element mesh containing geometry and connectivity
   * @param materials Material properties database indexed by element
   * @param has_gll_model If true, skips material property assignment (for GLL
   * models)
   *
   * @code
   * specfem::assembly::properties<specfem::element::dimension_tag::dim2> props(
   *     1000, 5, 5, element_types, mesh, materials, false);
   * @endcode
   */
  properties(
      const int nspec, const int ngllz, const int ngllx,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::mesh::materials<dimension_tag> &materials,
      bool has_gll_model);

  /**
   * @brief Copy material properties data from device to host memory.
   *
   * Transfers all material property data stored in device memory back to
   * host-accessible memory for post-processing, I/O operations, or debugging.
   */
  void copy_to_host() {
    impl::value_containers<
        dimension_tag,
        specfem::assembly::impl::domain_properties>::copy_to_host();
  }

  /**
   * @brief Copy material properties data from host to device memory.
   *
   * Transfers material property data from host memory to device-accessible
   * memory for use in GPU-accelerated computations.
   */
  void copy_to_device() {
    impl::value_containers<
        dimension_tag,
        specfem::assembly::impl::domain_properties>::copy_to_device();
  }
};

/**
 * @brief Compute maximum material property values across specified elements.
 *
 * This function performs a parallel reduction to find the maximum values
 * of material properties across a subset of spectral elements. It maps
 * element indices to property indices and delegates the reduction operation
 * to the appropriate material property container.
 *
 * @tparam IndexViewType Kokkos view type containing element indices
 * @tparam PointPropertiesType Material properties type (e.g., density, wave
 * speeds)
 *
 * @param ispecs View containing spectral element indices to process
 * @param properties Material properties container for the 2D mesh
 * @param point_properties Output container for maximum property values
 *
 * @code
 * // Find maximum density across elements 0-99
 * Kokkos::View<int*> elements("elements", 100);
 * specfem::point::properties<dim2, elastic, isotropic> max_props;
 * max(elements, assembly_props, max_props);
 * @endcode
 */
template <typename IndexViewType, typename PointPropertiesType>
void max(
    const IndexViewType &ispecs,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim2>
        &properties,
    PointPropertiesType &point_properties) {

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;

  constexpr bool on_device =
      std::is_same<typename IndexViewType::execution_space,
                   Kokkos::DefaultExecutionSpace>::value;

  static_assert(PointPropertiesType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
                "Only 2D properties are supported");

  IndexViewType local_ispecs("local_ispecs", ispecs.extent(0));

  const auto index_mapping = properties.get_property_index_mapping<on_device>();

  Kokkos::parallel_for(
      "local_work_items",
      Kokkos::RangePolicy<typename IndexViewType::execution_space>(
          0, ispecs.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        local_ispecs(i) =
            index_mapping(ispecs(i)); // Map the ispec to the property index
      });

  Kokkos::fence(); // Ensure the above parallel for is complete before
                   // proceeding

  properties.get_container<MediumTag, PropertyTag>().max(local_ispecs,
                                                         point_properties);

  // Note: The above call to max will perform the reduction on the device

  return;
}
} // namespace specfem::assembly
