#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/compute/impl/compute_coupling.hpp"
#include "specfem/compute/impl/compute_source_interaction.hpp"
#include "specfem/compute/impl/compute_stiffness_interaction.hpp"
#include "specfem/compute/impl/divide_mass_matrix.hpp"
#include "specfem/compute/impl/enforce_acoustic_free_surface.hpp"
#include "specfem/compute/initialize_mass_matrix.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/element.hpp"
#include "specfem/solver/mpi_buffers.hpp"
#include "specfem/tags.hpp"

namespace specfem::solver::impl {

/**
 * @brief Update one medium's wavefield, overlapping MPI exchange with compute.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @tparam Tags Compile-time tags (dimension, wavefield, medium)
 * @param assembly The assembly object containing the mesh and fields
 * @param mpi_buffers Solver-owned MPI buffers
 * @param istep Current time step
 * @return Number of elements updated
 */
template <int NGLL, typename Tags>
int update_medium(specfem::assembly::assembly<Tags::dimension_tag> &assembly,
                  specfem::solver::MPIBuffers<Tags::dimension_tag> &mpi_buffers,
                  const int istep) {
  constexpr auto outer = specfem::element::mpi_tag::outer;
  constexpr auto inner = specfem::element::mpi_tag::inner;
  constexpr auto acceleration =
      specfem::data_access::DataClassType::acceleration;

  specfem::compute::impl::compute_coupling<NGLL, Tags>(assembly);
  specfem::compute::impl::compute_source_interaction<NGLL, Tags>(assembly,
                                                                 istep);

  auto &field =
      assembly.fields.template get_simulation_field<Tags::wavefield_tag>();

  auto acceleration_buffer =
      mpi_buffers.template get<Tags>().template get<acceleration>();

  int elements_updated = specfem::compute::impl::compute_stiffness_interaction<
      NGLL, specfem::tags::expand<Tags, outer>>(assembly, istep);

  acceleration_buffer.begin_communicate(field);

  elements_updated += specfem::compute::impl::compute_stiffness_interaction<
      NGLL, specfem::tags::expand<Tags, inner>>(assembly, istep);

  acceleration_buffer.finish_communicate(field);

  specfem::compute::impl::divide_mass_matrix<NGLL, Tags>(assembly);

  if constexpr (Tags::medium_tag == specfem::element::medium_tag::acoustic) {
    specfem::compute::impl::enforce_acoustic_free_surface<NGLL, Tags>(assembly);
  }

  return elements_updated;
}

/**
 * @brief Initialize one medium's mass matrix, overlapping exchange with
 * compute.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @tparam Tags Compile-time tags (dimension, wavefield, medium)
 * @param assembly The assembly object containing the mesh and fields
 * @param mpi_buffers Solver-owned MPI buffers
 * @param dt Time step for the simulation
 */
template <int NGLL, typename Tags>
void init_medium_mass(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    specfem::solver::MPIBuffers<Tags::dimension_tag> &mpi_buffers,
    const type_real &dt) {
  constexpr auto outer = specfem::element::mpi_tag::outer;
  constexpr auto inner = specfem::element::mpi_tag::inner;
  constexpr auto mass_matrix = specfem::data_access::DataClassType::mass_matrix;

  specfem::compute::compute_mass_matrix<NGLL,
                                        specfem::tags::expand<Tags, outer>>(
      assembly, dt);

  auto &field =
      assembly.fields.template get_simulation_field<Tags::wavefield_tag>();

  auto mass_matrix_buffer =
      mpi_buffers.template get<Tags>().template get<mass_matrix>();

  mass_matrix_buffer.begin_communicate(field);

  specfem::compute::compute_mass_matrix<NGLL,
                                        specfem::tags::expand<Tags, inner>>(
      assembly, dt);

  mass_matrix_buffer.finish_communicate(field);

  specfem::compute::invert_mass<NGLL, Tags>(assembly);
}

} // namespace specfem::solver::impl
