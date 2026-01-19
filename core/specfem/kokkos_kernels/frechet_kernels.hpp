#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "impl/compute_material_derivatives.hpp"
#include "specfem/assembly.hpp"
#include "specfem/macros.hpp"

namespace specfem {
namespace kokkos_kernels {

/**
 * @brief Compute kernels used to compute Frechet derivatives.
 *
 * @tparam DimensionTag Dimension of the problem.
 * @tparam NGLL Number of GLL points.
 */
template <specfem::dimension::type DimensionTag, int NGLL>
class frechet_kernels {
public:
  constexpr static auto dimension_tag =
      DimensionTag; ///< Dimension of the problem.

  /**
   * @brief Constructor.
   *
   * @param assembly Assembly object.
   */
  frechet_kernels(const specfem::assembly::assembly<dimension_tag> &assembly)
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
          if constexpr (dimension_tag == _dimension_tag_) {
            impl::compute_material_derivatives<dimension_tag, NGLL,
                                               _medium_tag_, _property_tag_>(
                this->assembly, dt);
          }
        })
  }

private:
  /**
   * @brief SPECFEM++ assembly object containing mesh and simulation data
   *
   * Assembly object provides the computational kernels access to mesh
   * connectivity, element properties, and other necessary simulation data.
   */
  specfem::assembly::assembly<dimension_tag> assembly;
};
} // namespace kokkos_kernels
} // namespace specfem
