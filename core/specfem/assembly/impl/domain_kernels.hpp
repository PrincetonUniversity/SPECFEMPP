#pragma once

#include "domain_accessor.hpp"
#include "specfem/datatype/element_index_range.hpp"
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
          DimensionTag, domain_kernels<DimensionTag, MediumTag, PropertyTag>> {

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

  int base_ispec = 0; ///< First global element index for this type's range

  /// Default constructor for empty kernels container
  domain_kernels() = default;

  /**
   * @brief Construct kernels container for specified elements.
   *
   * Initializes kernels storage for the given spectral elements.
   * All kernel values are initialized to zero for accumulation.
   *
   * @param elements Contiguous range of global element indices for this type
   * @param grid Element grid configuration
   */
  domain_kernels(const specfem::datatype::ElementIndexRange &elements,
                 const specfem::mesh_entity::element_grid<dimension_tag> &grid)
      : base_type(elements.extent(0), grid) {
    base_ispec = elements.begin_index();
  }
};

} // namespace specfem::assembly::impl
