#pragma once

#include "impl/compute_seismograms.hpp"
#include "specfem/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/tags.hpp"

namespace specfem::compute {
/**
 * @brief Computes the seismograms for the simulation
 *
 * This function computes the seismograms for the simulation. It is
 * specialized for different medium types and properties.
 *
 * @tparam WavefieldType Type of the wavefield (e.g., elastic, acoustic)
 * @tparam DimensionTag Dimension tag (e.g., 2D, 3D)
 * @tparam NGLL Number of GLL points
 * @param assembly The assembly object containing the mesh and other
 * @param isig_step Time step for which the seismograms are computed
 */
template <int NGLL, typename Tags>
void compute_seismograms(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const int &isig_step) {
  constexpr auto DimensionTag = Tags::dimension_tag;
  constexpr auto WavefieldType = Tags::wavefield_tag;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2, DIM3),
       MEDIUM_TAG(ELASTIC, ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
       ATTENUATION_TAG(NONE)),
      {
        if constexpr (DimensionTag == _dimension_tag_) {
          impl::compute_seismograms<
              NGLL, specfem::tags::Tags<DimensionTag, WavefieldType,
                                        _medium_tag_, _property_tag_> >(
              assembly, isig_step);
        }
      })
}
} // namespace specfem::compute
