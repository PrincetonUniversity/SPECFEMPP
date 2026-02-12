#pragma once

#include "impl/compute_mass_matrix.hpp"
#include "impl/invert_mass_matrix.hpp"
#include "specfem/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/tags.hpp"

namespace specfem::compute {
/**
 * @brief Initializes the mass matrix for the simulation
 *
 * This function initializes the mass matrix for the simulation. It computes
 * the mass matrix and inverts it for different medium types.
 *
 * @tparam WavefieldType Type of the wavefield
 * @tparam DimensionTag Dimension tag
 * @tparam NGLL Number of GLL points
 * @param assembly The assembly object containing the mesh
 * @param dt Time step for the simulation
 */
template <specfem::simulation::field_type WavefieldType,
          specfem::element::dimension_tag DimensionTag, int NGLL>
void initialize_mass_matrix(specfem::assembly::assembly<DimensionTag> &assembly,
                            const type_real &dt) {
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2, DIM3),
       MEDIUM_TAG(ELASTIC, ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
      {
        if constexpr (DimensionTag == _dimension_tag_) {
          specfem::compute::impl::compute_mass_matrix<
              NGLL,
              specfem::tags::Tags<DimensionTag, WavefieldType, _medium_tag_,
                                  _property_tag_, _boundary_tag_> >(dt,
                                                                    assembly);
        }
      })

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2, DIM3),
       MEDIUM_TAG(ELASTIC, ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T)),
      {
        if constexpr (DimensionTag == _dimension_tag_) {
          specfem::compute::impl::invert_mass_matrix<
              specfem::tags::Tags<DimensionTag, WavefieldType, _medium_tag_> >(
              assembly);
        }
      })

  return;
}
} // namespace specfem::compute
