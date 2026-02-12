#include "update_wavefields.hpp"
#include "impl/compute_coupling.tpp"
#include "impl/compute_source_interaction.tpp"
#include "impl/compute_stiffness_interaction.tpp"
#include "impl/divide_mass_matrix.tpp"
#include "specfem/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/tags.hpp"

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
template <int NGLL, typename Tags>
int update_wavefields(specfem::assembly::assembly<Tags::dimension_tag> &assembly,
                      const int istep) {

  constexpr auto WavefieldType = Tags::wavefield_tag;
  constexpr auto DimensionTag = Tags::dimension_tag;
  constexpr auto MediumTag = Tags::medium_tag;

  int elements_updated = 0;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET),
       FLUX_SCHEME_TAG(NATURAL)),
      {
        constexpr auto self_medium = specfem::element_coupling::attributes<
            _dimension_tag_, _interface_tag_>::self_medium();
        if constexpr (DimensionTag == _dimension_tag_ &&
                      self_medium == MediumTag) {
          impl::compute_coupling<
              NGLL, NGLL,
              specfem::tags::Tags<_dimension_tag_, _connection_tag_,
                                  WavefieldType, _interface_tag_,
                                  _boundary_tag_, _flux_scheme_tag_> >(
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
          impl::compute_source_interaction<
              NGLL,
              specfem::tags::Tags<DimensionTag, WavefieldType, _medium_tag_,
                                  _property_tag_, _boundary_tag_> >(assembly,
                                                                    istep);
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
              NGLL,
              specfem::tags::Tags<DimensionTag, WavefieldType, _medium_tag_,
                                  _property_tag_, _boundary_tag_> >(assembly,
                                                                    istep);
        }
      })

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2, DIM3),
       MEDIUM_TAG(ELASTIC, ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T)),
      {
        if constexpr (DimensionTag == _dimension_tag_ &&
                      MediumTag == _medium_tag_) {
          impl::divide_mass_matrix<
              specfem::tags::Tags<DimensionTag, WavefieldType, _medium_tag_> >(
              assembly);
        }
      })

  return elements_updated;
}
} // namespace specfem::compute
