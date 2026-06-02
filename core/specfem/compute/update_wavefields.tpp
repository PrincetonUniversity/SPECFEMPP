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
 * @brief Updates the wavefield for a given medium
 *
 * This function updates the wavefield for a given medium type. It computes
 * the coupling, source interaction, stiffness interaction, and divides the
 * mass matrix. The function is specialized for different medium types and
 * properties.
 *
 * For 3D simulations, stiffness computation is split into two phases to
 * overlap MPI communication with computation:
 * 1. Compute stiffness on outer elements (touching MPI boundaries)
 * 2. Begin async MPI exchange of acceleration
 * 3. Compute stiffness on inner elements (overlaps with MPI transfer)
 * 4. Complete MPI exchange and accumulate received contributions
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

  if constexpr (Tags::dimension_tag ==
                specfem::element::dimension_tag::dim3) {
    // Phase 1: Stiffness on OUTER elements (touching MPI boundaries)
    specfem::tag_dispatch::for_each(
        specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
            specfem::tag_dispatch::medium_set<Tags::medium_tag>{} *
            PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
            ATTENUATION_SET(none, constant_isotropic) *
            BOUNDARY_SET(none, stacey, acoustic_free_surface,
                         composite_stacey_dirichlet) *
            MPI_SET(outer),
        [&]<typename ElementTags>() {
          elements_updated +=
              impl::compute_stiffness_interaction<
                  NGLL,
                  specfem::tags::expand<ElementTags, Tags::wavefield_tag>>(
                  assembly, istep);
        });

    // Get a mutable reference to the simulation field for MPI exchange
    auto &sim_field = [&]() -> auto & {
      if constexpr (Tags::wavefield_tag ==
                    specfem::simulation::field_type::forward)
        return assembly.fields.forward;
      else if constexpr (Tags::wavefield_tag ==
                         specfem::simulation::field_type::backward)
        return assembly.fields.backward;
      else
        return assembly.fields.adjoint;
    }();

    // Phase 2: Begin async MPI exchange of acceleration
    assembly.accel_buffers.template begin_exchange<Tags::wavefield_tag,
                                                   Tags::medium_tag>(sim_field);

    // Phase 3: Stiffness on INNER elements (overlaps with MPI transfer)
    specfem::tag_dispatch::for_each(
        specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
            specfem::tag_dispatch::medium_set<Tags::medium_tag>{} *
            PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
            ATTENUATION_SET(none, constant_isotropic) *
            BOUNDARY_SET(none, stacey, acoustic_free_surface,
                         composite_stacey_dirichlet) *
            MPI_SET(inner),
        [&]<typename ElementTags>() {
          elements_updated +=
              impl::compute_stiffness_interaction<
                  NGLL,
                  specfem::tags::expand<ElementTags, Tags::wavefield_tag>>(
                  assembly, istep);
        });

    // Phase 4: Complete MPI exchange and accumulate
    assembly.accel_buffers.template complete_exchange<Tags::wavefield_tag,
                                                     Tags::medium_tag>(
        sim_field);
  } else {
    // Dim2: no MPI overlap (existing behavior)
    specfem::tag_dispatch::for_each(
        specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
            specfem::tag_dispatch::medium_set<Tags::medium_tag>{} *
            PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
            ATTENUATION_SET(none, constant_isotropic) *
            BOUNDARY_SET(none, stacey, acoustic_free_surface,
                         composite_stacey_dirichlet),
        [&]<typename ElementTags>() {
          elements_updated +=
              impl::compute_stiffness_interaction<
                  NGLL,
                  specfem::tags::expand<ElementTags, Tags::wavefield_tag>>(
                  assembly, istep);
        });
  }

  impl::divide_mass_matrix<NGLL, Tags>(assembly);
  return elements_updated;
}

} // namespace specfem::compute
