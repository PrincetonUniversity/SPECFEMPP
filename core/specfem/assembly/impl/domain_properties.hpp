#pragma once

#include "domain_accessor.hpp"
#include "specfem/element.hpp"
#include "specfem/medium_container.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
class mesh_to_compute_mapping;
template <specfem::element::dimension_tag Dimension> struct mesh;
} // namespace specfem::assembly

namespace specfem::mesh {
template <specfem::element::dimension_tag Dimension> struct materials;
}

namespace specfem::assembly::impl {

/**
 * @brief Material properties storage for spectral elements at quadrature
 * points.
 *
 * Template container that stores material properties (density, elastic moduli,
 * etc.) for all quadrature points within spectral elements. Specializes for
 * different dimension/medium/property combinations and provides efficient
 * Kokkos-based storage.
 *
 * Inherits from `properties::data_container` for storage and `impl::Accessor`
 * for element-wise access. Material properties are organized as:
 * - **2D**: properties[nspec][ngllz][ngllx]
 * - **3D**: properties[nspec][ngllz][nglly][ngllx]
 *
 * Template parameters select appropriate specialization:
 * - **DimensionTag**: Spatial dimension (dim2/dim3)
 * - **MediumTag**: Physical medium (acoustic, elastic, poroelastic)
 * - **PropertyTag**: Material symmetry (isotropic, anisotropic)
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam MediumTag Physical medium type
 * @tparam PropertyTag Material property type
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct domain_properties;

/**
 * @brief 2D material properties container specialization.
 *
 * Stores material properties at quadrature points for 2D spectral elements.
 * Data layout: properties[element][ngllz][ngllx] where ngllz and ngllx are
 * quadrature points in vertical and horizontal directions.
 *
 * @tparam MediumTag Physical medium (acoustic, elastic, poroelastic)
 * @tparam PropertyTag Material symmetry (isotropic, anisotropic, cosserat)
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct domain_properties<specfem::element::dimension_tag::dim2, MediumTag,
                         PropertyTag>
    : public specfem::medium_container::properties::data_container<
          specfem::element::dimension_tag::dim2, MediumTag, PropertyTag>,
      public DomainAccessor<
          specfem::element::dimension_tag::dim2,
          domain_properties<specfem::element::dimension_tag::dim2, MediumTag,
                            PropertyTag> > {

  /// Base data container type for property storage
  using base_type = specfem::medium_container::properties::data_container<
      specfem::element::dimension_tag::dim2, MediumTag, PropertyTag>;
  using base_type::base_type;

  constexpr static auto dimension_tag =
      base_type::dimension_tag; ///< 2D spatial dimension
  constexpr static auto medium_tag =
      base_type::medium_tag; ///< Physical medium type
  constexpr static auto property_tag =
      base_type::property_tag; ///< Material property type

  /// Default constructor for empty container
  domain_properties() = default;

  /**
   * @brief Construct 2D properties from mesh and material data.
   *
   * Extracts material properties from mesh materials and stores them
   * at all quadrature points for the specified spectral elements.
   *
   * @param elements Element indices to initialize
   * @param mesh 2D mesh geometry and connectivity
   * @param ngllz Number of vertical quadrature points
   * @param ngllx Number of horizontal quadrature points
   * @param materials Material database indexed by element
   * @param has_gll_model Skip material assignment for GLL models
   * @param property_index_mapping Element to property mapping
   */
  template <specfem::element::attenuation_tag AttenuationTag>
  domain_properties(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const specfem::assembly::mesh<dimension_tag> &mesh, const int ngllz,
      const int ngllx, const specfem::mesh::materials<dimension_tag> &materials,
      const bool has_gll_model,
      const Kokkos::View<int *, Kokkos::LayoutRight,
                         Kokkos::DefaultHostExecutionSpace>
          property_index_mapping,
      std::integral_constant<specfem::element::attenuation_tag,
                             AttenuationTag>);

  /// Device value access disabled for this container type
  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_device_values(const IndexType &index, PointValues &values) const = delete;

  /// Host value access disabled for this container type
  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_host_values(const IndexType &index, PointValues &values) const = delete;
};

/**
 * @brief 3D material properties container specialization.
 *
 * Stores material properties at quadrature points for 3D spectral elements.
 * Data layout: properties[element][ngllz][nglly][ngllx] where ngllz, nglly,
 * and ngllx are quadrature points in z, y, and x directions respectively.
 *
 * @tparam MediumTag Physical medium (acoustic, elastic)
 * @tparam PropertyTag Material symmetry (isotropic, anisotropic)
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct domain_properties<specfem::element::dimension_tag::dim3, MediumTag,
                         PropertyTag>
    : public specfem::medium_container::properties::data_container<
          specfem::element::dimension_tag::dim3, MediumTag, PropertyTag>,
      public DomainAccessor<
          specfem::element::dimension_tag::dim3,
          domain_properties<specfem::element::dimension_tag::dim3, MediumTag,
                            PropertyTag> > {

  /// Base data container type for property storage
  using base_type = specfem::medium_container::properties::data_container<
      specfem::element::dimension_tag::dim3, MediumTag, PropertyTag>;
  using base_type::base_type;

  constexpr static auto dimension_tag =
      base_type::dimension_tag; ///< 3D spatial dimension
  constexpr static auto medium_tag =
      base_type::medium_tag; ///< Physical medium type
  constexpr static auto property_tag =
      base_type::property_tag; ///< Material property type

  /// Default constructor for empty container
  domain_properties() = default;

  /**
   * @brief Construct 3D properties from mesh and material data.
   *
   * Extracts material properties from mesh materials and stores them
   * at all quadrature points for the specified spectral elements.
   *
   * @param elements Element indices to initialize
   * @param nspec Total number of spectral elements
   * @param ngllz Number of z-direction quadrature points
   * @param nglly Number of y-direction quadrature points
   * @param ngllx Number of x-direction quadrature points
   * @param materials Material database indexed by element
   * @param property_index_mapping Element to property mapping
   */
  template <specfem::element::attenuation_tag AttenuationTag>
  domain_properties(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int nspec, const int ngllz, const int nglly, const int ngllx,
      const specfem::mesh::materials<dimension_tag> &materials,
      const Kokkos::View<int *, Kokkos::LayoutRight,
                         Kokkos::DefaultHostExecutionSpace>
          property_index_mapping,
      std::integral_constant<specfem::element::attenuation_tag,
                             AttenuationTag>);

  /// Device value access disabled for this container type
  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_device_values(const IndexType &index, PointValues &values) const = delete;

  /// Host value access disabled for this container type
  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_host_values(const IndexType &index, PointValues &values) const = delete;
};

} // namespace specfem::assembly::impl
