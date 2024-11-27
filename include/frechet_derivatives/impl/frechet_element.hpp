#ifndef _FRECHET_DERIVATIVES_IMPL_FRECHLET_ELEMENT_HPP
#define _FRECHET_DERIVATIVES_IMPL_FRECHLET_ELEMENT_HPP

#include "chunk_element/field.hpp"
#include "compute/assembly/assembly.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "policies/chunk.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace frechet_derivatives {
namespace impl {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
class KernelDataTypes {
public:
  constexpr static bool using_simd = true;
  using simd = specfem::datatype::simd<type_real, using_simd>;
  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      DimensionType, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkElementFieldType = specfem::chunk_element::field<
      ParallelConfig::chunk_size, NGLL, DimensionType, MediumTag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, false, false, false, using_simd>;

  using ElementQuadratureType = specfem::element::quadrature<
      NGLL, DimensionType, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

  using AdjointPointFieldType =
      specfem::point::field<DimensionType, MediumTag, false, false, true, false,
                            using_simd>;

  using BackwardPointFieldType =
      specfem::point::field<DimensionType, MediumTag, true, false, false, false,
                            using_simd>;

  using PointFieldDerivativesType =
      specfem::point::field_derivatives<DimensionType, MediumTag, using_simd>;

  using PointPropertiesType =
      specfem::point::properties<DimensionType, MediumTag, PropertyTag,
                                 using_simd>;

  using ChunkPolicy = specfem::policy::element_chunk<ParallelConfig>;
};

/**
 * @brief Compute Frechet derivatives for elements within a medium and property
 * tag
 *
 * @tparam DimensionType Dimension for elements within this kernel
 * @tparam MediumTag Medium for elements within this kernel
 * @tparam PropertyTag Property for elements within this kernel
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points for elements within this
 * kernel
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
class frechet_elements {
private:
  using datatype = KernelDataTypes<DimensionType, MediumTag, PropertyTag, NGLL>;
  using simd = typename datatype::simd;
  using ParallelConfig = typename datatype::ParallelConfig;
  using ChunkElementFieldType = typename datatype::ChunkElementFieldType;
  using ElementQuadratureType = typename datatype::ElementQuadratureType;
  using AdjointPointFieldType = typename datatype::AdjointPointFieldType;
  using BackwardPointFieldType = typename datatype::BackwardPointFieldType;
  using PointFieldDerivativesType =
      typename datatype::PointFieldDerivativesType;
  using PointPropertiesType = typename datatype::PointPropertiesType;
  using ChunkPolicy = typename datatype::ChunkPolicy;

public:
  /**
   * @name Constructor
   */
  ///@{

  /**
   * @brief Construct a Freclet Derivatives kernels from spectral element
   * assembly
   *
   * @param assembly Spectral element assembly
   */
  frechet_elements(const specfem::compute::assembly &assembly);
  ///@}

  /**
   * @brief Compute Frechet derivatives
   *
   * @param dt Time step
   */
  void compute(const type_real &dt);

private:
  specfem::kokkos::DeviceView1d<int> element_index; ///< Spectral element index
                                                    ///< for elements within
                                                    ///< this kernel
  specfem::kokkos::HostMirror1d<int> h_element_index; ///< Host mirror of
                                                      ///< element_index
  specfem::compute::simulation_field<
      specfem::wavefield::simulation_field::adjoint>
      adjoint_field; ///< Adjoint field
  specfem::compute::simulation_field<
      specfem::wavefield::simulation_field::backward>
      backward_field;                      ///< Backward field
  specfem::compute::kernels kernels;       ///< Misfit kernels
  specfem::compute::quadrature quadrature; ///< Integration quadrature
  specfem::compute::properties properties; ///< Material properties
  specfem::compute::partial_derivatives partial_derivatives; ///< Spatial
                                                             ///< derivatives of
                                                             ///< shape
                                                             ///< functions
};
} // namespace impl
} // namespace frechet_derivatives
} // namespace specfem

#endif /* _FRECHET_DERIVATIVES_IMPL_FRECHLET_ELEMENT_HPP */
