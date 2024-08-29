#pragma once

#include "chunk_element/field.hpp"
#include "chunk_element/stress_integrand.hpp"
#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "parallel_configuration/chunk_config.hpp"
#include "point/boundary.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress_integrand.hpp"
#include "policies/chunk.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {
/**
 * @brief Datatypes used in the kernels
 */
template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag, int NGLL>
class KernelDatatypes {
public:
  constexpr static auto wavefield_type = WavefieldType;
  constexpr static auto dimension = DimensionType;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;
  constexpr static auto boundary_tag = BoundaryTag;
  constexpr static int ngll = NGLL;
  constexpr static bool using_simd = true;

  using simd = specfem::datatype::simd<type_real, using_simd>;
  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      DimensionType, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkPolicyType = specfem::policy::element_chunk<ParallelConfig>;

  using PointBoundaryType =
      specfem::point::boundary<BoundaryTag, DimensionType, using_simd>;

  constexpr static int num_dimensions =
      specfem::dimension::dimension<dimension>::dim;
  constexpr static int components =
      specfem::medium::medium<dimension, medium_tag>::components;

  using ChunkElementFieldType = specfem::chunk_element::field<
      ParallelConfig::chunk_size, ngll, DimensionType, MediumTag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, false, false, false, using_simd>;

  using ChunkStressIntegrandType = specfem::chunk_element::stress_integrand<
      ParallelConfig::chunk_size, ngll, DimensionType, MediumTag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      using_simd>;

  using ElementQuadratureType = specfem::element::quadrature<
      ngll, specfem::dimension::type::dim2, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true>;

  using PointAccelerationType =
      specfem::point::field<DimensionType, MediumTag, false, false, true, false,
                            using_simd>;
  using PointVelocityType =
      specfem::point::field<DimensionType, MediumTag, false, true, false, false,
                            using_simd>;

  using PointFieldDerivativesType =
      specfem::point::field_derivatives<DimensionType, MediumTag, using_simd>;

  using PointMassType = specfem::point::field<DimensionType, MediumTag, false,
                                              false, false, true, using_simd>;

  using PointPropertyType =
      specfem::point::properties<dimension, medium_tag, property_tag,
                                 using_simd>;

  using PointPartialDerivativesType =
      specfem::point::partial_derivatives<dimension, true, using_simd>;
};

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag, int NGLL>
class element_kernel_base {
private:
  /// Datatypes used in the kernels
  using datatypes = KernelDatatypes<WavefieldType, DimensionType, MediumTag,
                                    PropertyTag, BoundaryTag, NGLL>;
  using simd = typename datatypes::simd;
  using ChunkPolicyType = typename datatypes::ChunkPolicyType;
  using PointBoundaryType = typename datatypes::PointBoundaryType;
  using ChunkElementFieldType = typename datatypes::ChunkElementFieldType;
  using ChunkStressIntegrandType = typename datatypes::ChunkStressIntegrandType;
  using ElementQuadratureType = typename datatypes::ElementQuadratureType;
  using PointAccelerationType = typename datatypes::PointAccelerationType;
  using PointVelocityType = typename datatypes::PointVelocityType;
  using PointFieldDerivativesType =
      typename datatypes::PointFieldDerivativesType;
  using PointMassType = typename datatypes::PointMassType;
  using PointPropertyType = typename datatypes::PointPropertyType;
  using PointPartialDerivativesType =
      typename datatypes::PointPartialDerivativesType;

  constexpr static int components = datatypes::components;
  constexpr static int num_dimensions = datatypes::num_dimensions;

public:
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto wavefield_type = WavefieldType; ///< Type of wavefield
  constexpr static auto dimension =
      DimensionType;                            ///< Dimension of the elements
  constexpr static auto medium_tag = MediumTag; ///< Medium tag of the elements
  constexpr static auto property_tag =
      PropertyTag; ///< Property tag of the elements
  constexpr static auto boundary_tag =
      BoundaryTag; ///< Boundary tag of the elements
  constexpr static int ngll = NGLL;
  ///@}

public:
  /**
   * @brief Get the total number of elements in this kernel
   *
   * @return int Number of elements
   */
  inline int total_elements() const { return nelements; }

  element_kernel_base() = default;
  element_kernel_base(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::HostView1d<int> h_element_kernel_index_mapping);

  void compute_mass_matrix(
      const type_real dt,
      const specfem::compute::simulation_field<WavefieldType> &field) const;

  void compute_stiffness_interaction(
      const int istep,
      const specfem::compute::simulation_field<WavefieldType> &field) const;

protected:
  int nelements;                   ///< Number of elements in this kernel
  specfem::compute::points points; ///< Assembly information
  specfem::compute::quadrature quadrature; ///< Information on integration
                                           ///< quadrature
  specfem::kokkos::DeviceView1d<int>
      element_kernel_index_mapping; ///< Spectral element index for every
                                    ///< element in this kernel
  specfem::kokkos::HostMirror1d<int>
      h_element_kernel_index_mapping;      ///< Host mirror of
                                           ///< element_kernel_index_mapping
  specfem::compute::properties properties; ///< Material properties
  specfem::compute::partial_derivatives partial_derivatives; ///< Spatial
                                                             ///< derivatives of
                                                             ///< basis
                                                             ///< functions
  specfem::compute::boundaries boundaries; ///< Boundary information
  specfem::compute::boundary_value_container<DimensionType, BoundaryTag>
      boundary_values; ///< Boundary values to store information on field values
                       ///< at boundaries for reconstruction during adjoint
                       ///< simulations
};

/**
 * @brief Compute Kernels for computing evolution of wavefield within elements
 * defined by element tags
 *
 * @tparam WavefieldType Type of the wavefield on which this kernel operates
 * @tparam DimensionType Dimension for the elements within this kernel
 * @tparam MediumTag Medium tag for the elements within this kernel
 * @tparam PropertyTag Property tag for the elements within this kernel
 * @tparam BoundaryTag Boundary tag for the elements within this kernel
 * @tparam NGLL Number of GLL points in each dimension for the elements within
 * this kernel
 */
template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag, int NGLL>
class element_kernel
    : public element_kernel_base<WavefieldType, DimensionType, MediumTag,
                                 PropertyTag, BoundaryTag, NGLL> {

public:
  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default Constructor
   */
  element_kernel() = default;

  /**
   * @brief Construct a element kernels based on assembly information
   *
   * @param assembly Assembly information
   * @param h_element_kernel_index_mapping Spectral element index for every
   * element in this kernel
   */
  element_kernel(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::HostView1d<int> h_element_kernel_index_mapping)
      : field(assembly.fields.get_simulation_field<WavefieldType>()),
        element_kernel_base<WavefieldType, DimensionType, MediumTag,
                            PropertyTag, BoundaryTag, NGLL>(
            assembly, h_element_kernel_index_mapping) {}
  ///@}

  /**
   * @brief Compute the mass matrix for the elements in this kernel
   *
   * @param dt Time step
   */
  void compute_mass_matrix(const type_real dt) const {
    element_kernel_base<WavefieldType, DimensionType, MediumTag, PropertyTag,
                        BoundaryTag, NGLL>::compute_mass_matrix(dt, field);
  }

  /**
   * @brief Compute the interaction of wavefield with stiffness matrix
   *
   * @param istep Time step
   */
  void compute_stiffness_interaction(const int istep) const {
    element_kernel_base<WavefieldType, DimensionType, MediumTag, PropertyTag,
                        BoundaryTag,
                        NGLL>::compute_stiffness_interaction(istep, field);
  }

private:
  specfem::compute::simulation_field<WavefieldType> field; ///< Wavefield
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
class element_kernel<specfem::wavefield::type::backward, DimensionType,
                     MediumTag, PropertyTag,
                     specfem::element::boundary_tag::stacey, NGLL>
    : public element_kernel_base<specfem::wavefield::type::backward,
                                 DimensionType, MediumTag, PropertyTag,
                                 specfem::element::boundary_tag::stacey, NGLL> {

public:
  element_kernel() = default;
  element_kernel(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::HostView1d<int> h_element_kernel_index_mapping)
      : field(assembly.fields
                  .get_simulation_field<specfem::wavefield::type::backward>()),
        element_kernel_base<specfem::wavefield::type::backward, DimensionType,
                            MediumTag, PropertyTag,
                            specfem::element::boundary_tag::stacey, NGLL>(
            assembly, h_element_kernel_index_mapping) {}

  void compute_mass_matrix(const type_real dt) const {
    element_kernel_base<specfem::wavefield::type::backward, DimensionType,
                        MediumTag, PropertyTag,
                        specfem::element::boundary_tag::stacey,
                        NGLL>::compute_mass_matrix(dt, field);
  }

  void compute_stiffness_interaction(const int istep) const {};

private:
  specfem::compute::simulation_field<specfem::wavefield::type::backward> field;
};

} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem
