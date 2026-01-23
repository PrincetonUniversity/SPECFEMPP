#pragma once

#include "enumerations/interface.hpp"
#include "impl/compute_coupling.hpp"
#include "impl/compute_source_interaction.hpp"
#include "impl/compute_stiffness_interaction.hpp"
#include "impl/divide_mass_matrix.hpp"
#include "specfem/assembly.hpp"

namespace specfem::compute {
/**
 * @brief Updates the wavefield for a given medium
 *
 * This function updates the wavefield for a given medium type. It computes
 * the coupling, source interaction, stiffness interaction, and divides the
 * mass matrix. The function is specialized for different medium types and
 * properties.
 *
 * @tparam WavefieldType Type of the wavefield
 * @tparam DimensionTag Dimension tag
 * @tparam NGLL Number of GLL points
 * @tparam MediumTag Medium for which the wacefield is updated
 * @param assembly The assembly object containing the mesh
 * @param istep Time step for which the wavefield is updated
 * @return int Number of elements updated
 */
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionTag, int NGLL,
          specfem::element::medium_tag MediumTag>
int update_wavefields(specfem::assembly::assembly<DimensionTag> &assembly,
                      const int istep) {

  int elements_updated = 0;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
      {
        constexpr auto self_medium =
            specfem::interface::attributes<_dimension_tag_,
                                           _interface_tag_>::self_medium();
        if constexpr (DimensionTag == _dimension_tag_ &&
                      self_medium == MediumTag) {
          impl::compute_coupling<_dimension_tag_, _connection_tag_,
                                 WavefieldType, NGLL, NGLL, _interface_tag_,
                                 _boundary_tag_,
                                 specfem::interface::flux_scheme_tag::natural>(
              assembly);
          // second ngll is the number of quadrature points on the mortar.
        }
      })

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2, DIM3),
       MEDIUM_TAG(ELASTIC, ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
      {
        if constexpr (DimensionTag == _dimension_tag_ &&
                      MediumTag == _medium_tag_) {
          impl::compute_source_interaction<DimensionTag, WavefieldType, NGLL,
                                           _medium_tag_, _property_tag_,
                                           _boundary_tag_>(assembly, istep);
        }
      })

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2, DIM3),
       MEDIUM_TAG(ELASTIC, ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
      {
        if constexpr (DimensionTag == _dimension_tag_ &&
                      MediumTag == _medium_tag_) {
          elements_updated += impl::compute_stiffness_interaction<
              DimensionTag, WavefieldType, NGLL, _medium_tag_, _property_tag_,
              _boundary_tag_>(assembly, istep);
        }
      })

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2, DIM3),
       MEDIUM_TAG(ELASTIC, ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T)),
      {
        if constexpr (DimensionTag == _dimension_tag_ &&
                      MediumTag == _medium_tag_) {
          impl::divide_mass_matrix<DimensionTag, WavefieldType, _medium_tag_>(
              assembly);
        }
      })

  return elements_updated;
}
} // namespace specfem::compute
