#pragma once

#include "domain_accessor.hpp"
#include "specfem/medium_container.hpp"
#include "specfem/mesh_entity.hpp"

namespace specfem::assembly::impl {

/**
 * @brief Misfit kernel storage container for seismic inversion.
 *
 * Template container that stores sensitivity kernels (Frechet derivatives)
 * which represent the gradient of
 * the misfit function with respect to material parameters and are computed
 * from the interaction of forward and adjoint wavefields.
 *
 * Specializes for different dimension/medium/property combinations and provides
 * efficient accumulation operations for kernel computation during adjoint
 * simulations. Inherits from `kernels::data_container` for storage and
 * `specfem::assembly::impl::DomainAccessor` for device/host data access.
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 * @tparam MediumTag Physical medium type
 * @tparam PropertyTag Material property type
 *
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct domain_kernels
    : public specfem::medium_container::kernels::data_container<
          DimensionTag, MediumTag, PropertyTag>,
      public DomainAccessor<
          DimensionTag, domain_kernels<DimensionTag, MediumTag, PropertyTag> > {

  /// Base kernels data container type
  using base_type = specfem::medium_container::kernels::data_container<
      DimensionTag, MediumTag, PropertyTag>;
  using base_type::base_type;

  constexpr static auto dimension_tag =
      base_type::dimension_tag; ///< Spatial dimension
  constexpr static auto medium_tag =
      base_type::medium_tag; ///< Physical medium type
  constexpr static auto property_tag =
      base_type::property_tag; ///< Material property type

  /// Default constructor for empty kernels container
  domain_kernels() = default;

  /**
   * @brief Construct kernels container for specified elements.
   *
   * Initializes kernels storage for the given spectral elements and sets up
   * the mapping from global element indices to local kernel storage indices.
   * All kernel values are initialized to zero for accumulation.
   *
   * @param elements Element indices to initialize kernels for
   * @param grid Element grid configuration
   * @param property_index_mapping Output mapping from element index to kernel
   * storage index
   */
  domain_kernels(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const specfem::mesh_entity::element_grid<dimension_tag> &grid,
      const Kokkos::View<int *, Kokkos::LayoutRight,
                         Kokkos::DefaultHostExecutionSpace>
          property_index_mapping)
      : base_type(elements.extent(0), grid) {
    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      property_index_mapping(ispec) = count;
      count++;
    }
  }
};

} // namespace specfem::assembly::impl
