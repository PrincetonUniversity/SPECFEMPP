#pragma once

#include "domain_accessor.hpp"
#include "specfem/medium_container.hpp"

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
          specfem::element::property_tag PropertyTag,
          specfem::element::attenuation_tag AttenuationTag =
              specfem::element::attenuation_tag::none>
struct domain_kernels;

/**
 * @brief 2D misfit kernels container specialization.
 *
 * Stores sensitivity kernels for 2D spectral elements at all quadrature
 * points. Data layout: kernels[element][ngllz][ngllx] where ngllz and ngllx
 * are quadrature points in vertical and horizontal directions.
 *
 * Used for accumulating kernel contributions during adjoint simulation:
 * - Forward wavefield provides displacement/velocity/acceleration
 * - Adjoint wavefield provides sensitivity information
 * - Kernels accumulate the correlation between these fields
 *
 * @tparam MediumTag Physical medium (acoustic, elastic, poroelastic)
 * @tparam PropertyTag Material symmetry (isotropic, anisotropic, cosserat)
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::attenuation_tag AttenuationTag>
struct domain_kernels<specfem::element::dimension_tag::dim2, MediumTag,
                      PropertyTag, AttenuationTag>
    : public specfem::medium_container::kernels::data_container<
          specfem::element::dimension_tag::dim2, MediumTag, PropertyTag>,
      public DomainAccessor<
          specfem::element::dimension_tag::dim2,
          domain_kernels<specfem::element::dimension_tag::dim2, MediumTag,
                         PropertyTag, AttenuationTag> > {

  /// Base kernels data container type
  using base_type = specfem::medium_container::kernels::data_container<
      specfem::element::dimension_tag::dim2, MediumTag, PropertyTag>;
  using base_type::base_type;

  constexpr static auto dimension_tag =
      base_type::dimension_tag; ///< 2D spatial dimension
  constexpr static auto medium_tag =
      base_type::medium_tag; ///< Physical medium type
  constexpr static auto property_tag =
      base_type::property_tag; ///< Material property type

  /// Default constructor for empty kernels container
  domain_kernels() = default;

  /**
   * @brief Construct 2D kernels container for specified elements.
   *
   * Initializes kernels storage for the given spectral elements and sets up
   * the mapping from global element indices to local kernel storage indices.
   * All kernel values are initialized to zero for accumulation.
   *
   * @param elements Element indices to initialize kernels for
   * @param ngllz Number of vertical quadrature points per element
   * @param ngllx Number of horizontal quadrature points per element
   * @param property_index_mapping Output mapping from element index to kernel
   * storage index
   */
  domain_kernels(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int ngllx,
      const Kokkos::View<int *, Kokkos::LayoutRight,
                         Kokkos::DefaultHostExecutionSpace>
          property_index_mapping)
      : base_type(elements.extent(0), ngllz, ngllx) {
    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      property_index_mapping(ispec) = count;
      count++;
    }
  }
};

/**
 * @brief 3D misfit kernels container specialization.
 *
 * Stores sensitivity kernels for 3D spectral elements at all quadrature
 * points. Data layout: kernels[element][ngllz][nglly][ngllx] where ngllz,
 * nglly, and ngllx are quadrature points in z, y, and x directions.
 *
 * Used for accumulating kernel contributions during 3D adjoint simulation.
 * Provides efficient storage
 * and access patterns for 3D kernel accumulation operations.
 *
 * @tparam MediumTag Physical medium (acoustic, elastic)
 * @tparam PropertyTag Material symmetry (isotropic, anisotropic)
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::attenuation_tag AttenuationTag>
struct domain_kernels<specfem::element::dimension_tag::dim3, MediumTag,
                      PropertyTag, AttenuationTag>
    : public specfem::medium_container::kernels::data_container<
          specfem::element::dimension_tag::dim3, MediumTag, PropertyTag>,
      public DomainAccessor<
          specfem::element::dimension_tag::dim3,
          domain_kernels<specfem::element::dimension_tag::dim3, MediumTag,
                         PropertyTag, AttenuationTag> > {

  /// Base kernels data container type
  using base_type = specfem::medium_container::kernels::data_container<
      specfem::element::dimension_tag::dim3, MediumTag, PropertyTag>;
  using base_type::base_type;

  constexpr static auto dimension_tag =
      base_type::dimension_tag; ///< 3D spatial dimension
  constexpr static auto medium_tag =
      base_type::medium_tag; ///< Physical medium type
  constexpr static auto property_tag =
      base_type::property_tag; ///< Material property type

  /// Default constructor for empty kernels container
  domain_kernels() = default;

  /**
   * @brief Construct 3D kernels container for specified elements.
   *
   * Initializes kernels storage for the given spectral elements and sets up
   * the mapping from global element indices to local kernel storage indices.
   * All kernel values are initialized to zero for accumulation.
   *
   * @param elements Element indices to initialize kernels for
   * @param ngllz Number of z-direction quadrature points per element
   * @param nglly Number of y-direction quadrature points per element
   * @param ngllx Number of x-direction quadrature points per element
   * @param property_index_mapping Output mapping from element index to kernel
   * storage index
   */
  domain_kernels(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int nglly, const int ngllx,
      const Kokkos::View<int *, Kokkos::LayoutRight,
                         Kokkos::DefaultHostExecutionSpace>
          property_index_mapping)
      : base_type(elements.extent(0), ngllz, nglly, ngllx) {
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
