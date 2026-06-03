#include "impl/compute_coupling.hpp"
#include "impl/compute_source_interaction.hpp"
#include "impl/compute_stiffness_interaction.hpp"
#include "impl/divide_mass_matrix.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "update_wavefields.hpp"

namespace specfem::compute {

/**
 * @brief Updates the wavefield for a given medium.
 *
 * Computes coupling, source interaction, stiffness interaction (with MPI
 * communication-computation overlap), and divides by the mass matrix.
 *
 * @tparam NGLL Number of GLL points
 * @tparam Tags Compile-time tags (dimension, medium, wavefield, etc.)
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

  const int elements_updated =
      impl::compute_stiffness_interaction<NGLL, Tags>(assembly, istep);

  impl::divide_mass_matrix<NGLL, Tags>(assembly);
  return elements_updated;
}

} // namespace specfem::compute
