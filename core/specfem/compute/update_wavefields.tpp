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

  int elements_updated = 0;

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
      specfem::tag_dispatch::medium_set<Tags::medium_tag>{} *
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
      ATTENUATION_SET(none) *
      BOUNDARY_SET(none, stacey, acoustic_free_surface,
                   composite_stacey_dirichlet), [&]<typename ElementTags>() {
    elements_updated +=
        impl::compute_stiffness_interaction<NGLL, Tags::wavefield_tag, ElementTags>(
            assembly, istep);
  });

  impl::divide_mass_matrix<NGLL, Tags>(assembly);
  return elements_updated;
}

} // namespace specfem::compute
