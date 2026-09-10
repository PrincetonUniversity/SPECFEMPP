#pragma once

// The impl `.tpp` definitions are pulled into this header (not just the `.hpp`
// declarations) so the header-only `compute_mass_matrix` / `invert_mass`
// templates below are fully defined for callers that include only this header
// and have no explicit-instantiation translation unit to link against.
#include "impl/compute_mass_matrix.hpp"
#include "impl/compute_mass_matrix.tpp"
#include "impl/invert_mass_matrix.hpp"
#include "impl/invert_mass_matrix.tpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::compute {

/**
 * @brief Compute local mass-matrix contributions for one medium.
 *
 * Accumulates the local (per-rank) mass-matrix contributions for the medium and
 * wavefield identified by @p Tags, iterating over all property/attenuation/
 * boundary combinations. When @p Tags carries an `mpi_tag`, only the matching
 * (inner/outer) element subset is computed, enabling communication-computation
 * overlap; otherwise all elements are computed.
 *
 * @tparam NGLL Number of GLL points
 * @tparam Tags Compile-time tags (dimension, wavefield, medium[, mpi_tag])
 * @param assembly The assembly object containing the mesh and fields
 * @param dt Time step for the simulation
 */
template <int NGLL, typename Tags>
void compute_mass_matrix(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const type_real &dt) {
  constexpr auto wavefield_tag = Tags::wavefield_tag;
  constexpr auto dimension_tag = Tags::dimension_tag;
  constexpr auto medium_tag = Tags::medium_tag;
  constexpr auto mpi_tag = Tags::mpi_tag;

  constexpr auto property_set = []() {
    if constexpr (Tags::dimension_tag ==
                  specfem::element::dimension_tag::dim3) {
      return PROPERTY_SET(isotropic);
    } else {
      return PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat);
    }
  }();

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<dimension_tag>{} *
          specfem::tag_dispatch::medium_set<medium_tag>{} * property_set *
          ATTENUATION_SET(none, constant_isotropic) *
          BOUNDARY_SET(none, acoustic_free_surface, stacey,
                       composite_stacey_dirichlet) *
          specfem::tag_dispatch::wavefield_set<wavefield_tag>{} *
          specfem::tag_dispatch::mpi_set<mpi_tag>{},
      [&]<typename ElementTags>() {
        specfem::compute::impl::compute_mass_matrix<NGLL, ElementTags>(
            dt, assembly);
      });
}

/**
 * @brief Invert the assembled mass matrix for one medium.
 *
 * @tparam NGLL Number of GLL points (unused, kept for call-site symmetry)
 * @tparam Tags Compile-time tags (dimension, wavefield, medium)
 * @param assembly The assembly object containing the mesh and fields
 */
template <int NGLL, typename Tags>
void invert_mass(specfem::assembly::assembly<Tags::dimension_tag> &assembly) {
  specfem::compute::impl::invert_mass_matrix<specfem::tags::Tags<
      Tags::dimension_tag, Tags::wavefield_tag, Tags::medium_tag>>(assembly);
}

} // namespace specfem::compute
