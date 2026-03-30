#include "impl/compute_coupling.hpp"
#include "impl/compute_source_interaction.hpp"
#include "impl/compute_stiffness_interaction.hpp"
#include "impl/divide_mass_matrix.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/tags.hpp"
#include "update_wavefields.hpp"

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
int update_wavefields(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const int istep) {
  impl::compute_coupling<NGLL, Tags>(assembly);
  impl::compute_source_interaction<NGLL, Tags>(assembly, istep);
  const int n =
      impl::compute_stiffness_interaction<NGLL, Tags>(assembly, istep);
  impl::divide_mass_matrix<NGLL, Tags>(assembly);
  return n;
}

} // namespace specfem::compute
