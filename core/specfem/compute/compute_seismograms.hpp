#pragma once

#include "enumerations/interface.hpp"
#include "impl/compute_seismograms.hpp"
#include "specfem/assembly.hpp"

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
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionTag, int NGLL>
void compute_seismograms(specfem::assembly::assembly<DimensionTag> &assembly,
                         const int &isig_step) {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2, DIM3),
       MEDIUM_TAG(ELASTIC, ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      {
        if constexpr (DimensionTag == _dimension_tag_) {
          impl::compute_seismograms<WavefieldType, DimensionTag, NGLL,
                                    _medium_tag_, _property_tag_>(assembly,
                                                                  isig_step);
        }
      })
}
} // namespace specfem::compute
