#ifndef _SPECFEM_KERNELS_FRECHET_KERNELS_HPP
#define _SPECFEM_KERNELS_FRECHET_KERNELS_HPP

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "impl/compute_material_derivatives.hpp"

namespace specfem {
namespace kokkos_kernels {

/**
 * @brief Compute kernels used to compute Frechet derivatives.
 *
 * @tparam DimensionType Dimension of the problem.
 * @tparam NGLL Number of GLL points.
 */
template <specfem::dimension::type DimensionType, int NGLL>
class frechet_kernels {
public:
  constexpr static auto dimension =
      DimensionType; ///< Dimension of the problem.

  /**
   * @brief Constructor.
   *
   * @param assembly Assembly object.
   */
  frechet_kernels(const specfem::compute::assembly &assembly)
      : assembly(assembly) {}

  /**
   * @brief Compute the frechet derivatives at the current time step.
   *
   * @param dt Time interval.
   */
  inline void compute_derivatives(const type_real &dt) {
#define CALL_COMPUTE_MATERIAL_DERIVATIVES(DIMENSION_TAG, MEDIUM_TAG,           \
                                          PROPERTY_TAG)                        \
  if constexpr (dimension == GET_TAG(DIMENSION_TAG)) {                         \
    impl::compute_material_derivatives<                                        \
        DimensionType, NGLL, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(      \
        this->assembly, dt);                                                   \
  }

    CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
        CALL_COMPUTE_MATERIAL_DERIVATIVES,
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
                WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef CALL_COMPUTE_MATERIAL_DERIVATIVES
  }

private:
  specfem::compute::assembly assembly; ///< Assembly object.
};
} // namespace kokkos_kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_FRECHET_KERNELS_HPP */
