#pragma once

#include "impl/compute_mass_matrix.hpp"
#include "impl/invert_mass_matrix.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"

namespace specfem::compute {
/**
 * @brief Initializes the mass matrix for the simulation
 *
 * This function initializes the mass matrix for the simulation. It computes
 * the mass matrix and inverts it for different medium types.
 *
 * @tparam WavefieldType Type of the wavefield
 * @tparam DimensionTag Dimension tag
 * @tparam NGLL Number of GLL points
 * @param assembly The assembly object containing the mesh
 * @param dt Time step for the simulation
 */
template <int NGLL, typename Tags>
void initialize_mass_matrix(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const type_real &dt) {
  constexpr auto WavefieldType = Tags::wavefield_tag;

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          MEDIUM_SET(elastic, elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t, elastic_spin) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
          BOUNDARY_SET(none, acoustic_free_surface, stacey,
                       composite_stacey_dirichlet),
      [&]<typename ElementTags>() {
        specfem::compute::impl::compute_mass_matrix<
            NGLL,
            specfem::tags::Tags<
                Tags::dimension_tag, WavefieldType, ElementTags::medium_tag,
                ElementTags::property_tag, ElementTags::boundary_tag> >(
            dt, assembly);
      });

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          MEDIUM_SET(elastic, elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t, elastic_spin),
      [&]<typename ElementTags>() {
        specfem::compute::impl::invert_mass_matrix<specfem::tags::Tags<
            Tags::dimension_tag, WavefieldType, ElementTags::medium_tag> >(
            assembly);
      });

  return;
}
} // namespace specfem::compute
