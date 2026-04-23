#pragma once

#include "impl/compute_seismograms.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/tag_dispatch.hpp"
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

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          MEDIUM_SET(elastic, elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
      [&]<typename ElementTags>() {
        impl::compute_seismograms<
            NGLL, specfem::tags::Tags<Tags::dimension_tag, WavefieldType,
                                      ElementTags::medium_tag,
                                      ElementTags::property_tag> >(assembly,
                                                                   isig_step);
      });
}
} // namespace specfem::compute
