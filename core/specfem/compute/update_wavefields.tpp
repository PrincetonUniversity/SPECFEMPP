#include "impl/compute_coupling.hpp"
#include "impl/compute_source_interaction.hpp"
#include "impl/compute_stiffness_interaction.hpp"
#include "impl/divide_mass_matrix.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include "update_wavefields.hpp"

namespace specfem::compute {

/**
 * @brief Updates the wavefield for a given medium.
 *
 * Computes coupling, source interaction, stiffness interaction (with MPI
 * communication-computation overlap), and divides by the mass matrix.
 *
 * Stiffness is split into two phases to overlap MPI halo exchange:
 * 1. Compute stiffness on outer elements (touching MPI boundaries)
 * 2. Begin async MPI exchange of acceleration
 * 3. Compute stiffness on inner elements (overlaps with MPI transfer)
 * 4. Complete MPI exchange and accumulate received contributions
 *
 * For dim2 (no MPI yet), all elements are classified as inner, the outer
 * pass finds zero elements, and begin/complete_exchange are no-ops.
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

  int elements_updated = 0;

  // Tag set products for stiffness dispatch
  constexpr auto base_set =
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
      specfem::tag_dispatch::medium_set<Tags::medium_tag>{} *
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
      ATTENUATION_SET(none, constant_isotropic) *
      BOUNDARY_SET(none, stacey, acoustic_free_surface,
                   composite_stacey_dirichlet);
  constexpr auto outer_set = base_set * MPI_SET(outer);
  constexpr auto inner_set = base_set * MPI_SET(inner);

  auto stiffness = [&]<typename ElementTags>() {
    elements_updated +=
        impl::compute_stiffness_interaction<
            NGLL,
            specfem::tags::expand<ElementTags, Tags::wavefield_tag>>(
            assembly, istep);
  };

  // Phase 1: outer elements (touching MPI boundaries)
  specfem::tag_dispatch::for_each(outer_set, stiffness);

  // Phase 2: begin async MPI exchange of acceleration
  auto &sim_field =
      assembly.fields.template get_simulation_field<Tags::wavefield_tag>();
  assembly.accel_buffers.template begin_exchange<Tags::wavefield_tag,
                                                  Tags::medium_tag>(sim_field);

  // Phase 3: inner elements (overlaps with MPI transfer)
  specfem::tag_dispatch::for_each(inner_set, stiffness);

  // Phase 4: complete MPI exchange and accumulate
  assembly.accel_buffers.template complete_exchange<Tags::wavefield_tag,
                                                    Tags::medium_tag>(
      sim_field);

  impl::divide_mass_matrix<NGLL, Tags>(assembly);
  return elements_updated;
}

} // namespace specfem::compute
