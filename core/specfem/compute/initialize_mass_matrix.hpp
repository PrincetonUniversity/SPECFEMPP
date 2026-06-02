#pragma once

#include "impl/compute_mass_matrix.hpp"
#include "impl/invert_mass_matrix.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::compute {
/**
 * @brief Initializes the mass matrix for the simulation
 *
 * This function initializes the mass matrix for the simulation. It computes
 * the mass matrix, assembles shared boundary contributions across MPI ranks,
 * and inverts the mass matrix for different medium types.
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

  // Step 1: Compute local mass matrix contributions
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          MEDIUM_SET(elastic, elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
          BOUNDARY_SET(none, acoustic_free_surface, stacey,
                       composite_stacey_dirichlet),
      [&]<typename ElementTags>() {
        specfem::compute::impl::compute_mass_matrix<
            NGLL,
            specfem::tags::Tags<
                Tags::dimension_tag, WavefieldType, ElementTags::medium_tag,
                ElementTags::property_tag, ElementTags::boundary_tag>>(
            dt, assembly);
      });

  // Step 2: MPI assembly of mass matrix across partition boundaries
  if constexpr (Tags::dimension_tag == specfem::element::dimension_tag::dim3) {

    constexpr auto MassDataClassType =
        specfem::data_access::DataClassType::mass_matrix;
    specfem::tag_dispatch::for_each(
        specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
            MEDIUM_SET(elastic, acoustic),
        [&]<typename ElementTags>() {
          auto mpi_buf = assembly.mpi_interfaces.template create_mpi_buffer<
              WavefieldType, ElementTags::medium_tag, MassDataClassType>();

          auto &field = [&]() -> auto & {
            if constexpr (WavefieldType ==
                          specfem::simulation::field_type::forward)
              return assembly.fields.forward;
            else if constexpr (WavefieldType ==
                               specfem::simulation::field_type::adjoint)
              return assembly.fields.adjoint;
            else
              return assembly.fields.backward;
          }();

          mpi_buf.pack(field);
          mpi_buf.receive();
          mpi_buf.send();
          mpi_buf.wait();
          mpi_buf.unpack(field);
        });
    Kokkos::fence("initialize_mass_matrix::mpi_assembly");
  }

  // Step 3: Invert mass matrix
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          MEDIUM_SET(elastic, elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t),
      [&]<typename ElementTags>() {
        specfem::compute::impl::invert_mass_matrix<specfem::tags::Tags<
            Tags::dimension_tag, WavefieldType, ElementTags::medium_tag>>(
            assembly);
      });

  return;
}
} // namespace specfem::compute
