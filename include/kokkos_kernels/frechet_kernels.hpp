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
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
        {
          if constexpr (dimension == _dimension_tag_) {
            impl::compute_material_derivatives<dimension, NGLL, _medium_tag_,
                                               _property_tag_>(this->assembly,
                                                               dt);
          }
        })
  }

private:
  specfem::compute::assembly assembly; ///< Assembly object.
};
} // namespace kokkos_kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_FRECHET_KERNELS_HPP */
