#pragma once

#include "specfem/element.hpp"
#include "specfem/logger.hpp"
#include "specfem/simulation.hpp"
#include "specfem/solver/impl/update_medium.hpp"
#include "specfem/tags.hpp"
#include "specfem/timescheme.hpp"
#include <sstream>
#include <stdexcept>
#include <utility>

namespace specfem::solver::impl {

// Medium tags shared by the stepping helpers below. `constexpr` gives these
// internal linkage, so they are safe to define at namespace scope in a header.
constexpr auto acoustic = specfem::element::medium_tag::acoustic;
constexpr auto elastic = specfem::element::medium_tag::elastic;
constexpr auto elastic_psv = specfem::element::medium_tag::elastic_psv;
constexpr auto elastic_sh = specfem::element::medium_tag::elastic_sh;
constexpr auto poroelastic = specfem::element::medium_tag::poroelastic;
constexpr auto elastic_psv_t = specfem::element::medium_tag::elastic_psv_t;

/**
 * @brief Log periodic time-marching progress.
 *
 * @param istep Current time step
 * @param nstep Total number of time steps
 * @param log_every Emit a log message every this many steps
 */
inline void log_time_marching_progress(const int istep, const int nstep,
                                       const int log_every = 10) {
  if (istep % log_every == 0) {
    std::ostringstream message;
    message << "Progress : executed " << istep << " steps of " << nstep
            << " steps";
    specfem::Logger::info(message.str());
  }
}

template <specfem::element::dimension_tag DimensionTag>
void check_medium_step_update(
    specfem::assembly::assembly<DimensionTag> &assembly, const int dofs_updated,
    const int elements_updated, const char *update_name) {
  const int total_dof_to_be_updated =
      2 * assembly.get_total_degrees_of_freedom();
  const int total_elements_to_be_updated =
      assembly.get_total_number_of_elements();

  if (dofs_updated != total_dof_to_be_updated) {
    std::ostringstream message;
    message << "The " << update_name
            << " step has not updated all the degrees of freedom. "
            << "Only " << dofs_updated << " out of " << total_dof_to_be_updated
            << " degrees of freedom have been updated.";

    throw std::runtime_error(message.str());
  }

  if (elements_updated != total_elements_to_be_updated) {
    std::ostringstream message;
    message << "The " << update_name
            << " step has not updated all the elements. "
            << "Only " << elements_updated << " out of "
            << total_elements_to_be_updated << " elements have been updated.";

    throw std::runtime_error(message.str());
  }
}

/**
 * @brief Advance a single wavefield through one forward-style
 * predictor/corrector step across all media, honoring the coupling order
 * between acoustic, elastic, and poroelastic media.
 *
 * Used both for genuine forward simulations and for the forward-physics
 * replay pass of the undo-attenuation adjoint solver, since both drive
 * @c fields.forward through the same predictor/corrector sequence.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @tparam FieldType Wavefield tag to update (e.g. forward)
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @param time_scheme Time integration scheme driving the predictor/corrector
 * phases
 * @param assembly The assembly object containing the mesh and fields
 * @param mpi_buffers Solver-owned MPI buffers
 * @param istep Current time step
 * @return {degrees of freedom updated, elements updated}
 */
template <int NGLL, specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag>
std::pair<int, int>
apply_forward_step(specfem::time_scheme::time_scheme &time_scheme,
                   specfem::assembly::assembly<DimensionTag> &assembly,
                   specfem::solver::MPIBuffers<DimensionTag> &mpi_buffers,
                   const int istep) {

  int dofs_updated = 0;
  int elements_updated = 0;

  dofs_updated += time_scheme.apply_predictor_phase_forward(acoustic);
  dofs_updated += time_scheme.apply_predictor_phase_forward(elastic);
  dofs_updated += time_scheme.apply_predictor_phase_forward(elastic_psv);
  dofs_updated += time_scheme.apply_predictor_phase_forward(elastic_sh);
  dofs_updated += time_scheme.apply_predictor_phase_forward(poroelastic);
  dofs_updated += time_scheme.apply_predictor_phase_forward(elastic_psv_t);

  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, acoustic>>(
          assembly, mpi_buffers, istep);
  dofs_updated += time_scheme.apply_corrector_phase_forward(acoustic);

  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, elastic>>(
          assembly, mpi_buffers, istep);
  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, elastic_psv>>(
          assembly, mpi_buffers, istep);
  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, elastic_sh>>(
          assembly, mpi_buffers, istep);
  elements_updated += update_medium<
      NGLL, specfem::tags::Tags<DimensionTag, FieldType, elastic_psv_t>>(
      assembly, mpi_buffers, istep);
  dofs_updated += time_scheme.apply_corrector_phase_forward(elastic);
  dofs_updated += time_scheme.apply_corrector_phase_forward(elastic_psv);
  dofs_updated += time_scheme.apply_corrector_phase_forward(elastic_sh);
  dofs_updated += time_scheme.apply_corrector_phase_forward(elastic_psv_t);

  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, poroelastic>>(
          assembly, mpi_buffers, istep);
  dofs_updated += time_scheme.apply_corrector_phase_forward(poroelastic);

  check_medium_step_update(assembly, dofs_updated, elements_updated, "forward");

  return { dofs_updated, elements_updated };
}

/**
 * @brief Advance the adjoint wavefield through one predictor/corrector step
 * across all media, honoring the acoustic/elastic/poroelastic coupling order.
 *
 * Shared by the @c combined and @c combined_undoatt solvers. Unlike the
 * forward step this excludes @c elastic_psv_t, which the combined solvers do
 * not support.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @tparam FieldType Wavefield tag to update (adjoint)
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @param time_scheme Time integration scheme driving the phases
 * @param assembly The assembly object containing the mesh and fields
 * @param mpi_buffers Solver-owned MPI buffers
 * @param istep Current time step
 * @return {degrees of freedom updated, elements updated}
 */
template <int NGLL, specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag>
std::pair<int, int>
apply_adjoint_step(specfem::time_scheme::time_scheme &time_scheme,
                   specfem::assembly::assembly<DimensionTag> &assembly,
                   specfem::solver::MPIBuffers<DimensionTag> &mpi_buffers,
                   const int istep) {

  int dofs_updated = 0;
  int elements_updated = 0;

  dofs_updated += time_scheme.apply_predictor_phase_adjoint(acoustic);
  dofs_updated += time_scheme.apply_predictor_phase_adjoint(elastic);
  dofs_updated += time_scheme.apply_predictor_phase_adjoint(elastic_psv);
  dofs_updated += time_scheme.apply_predictor_phase_adjoint(elastic_sh);
  dofs_updated += time_scheme.apply_predictor_phase_adjoint(poroelastic);

  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, acoustic>>(
          assembly, mpi_buffers, istep);
  dofs_updated += time_scheme.apply_corrector_phase_adjoint(acoustic);

  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, elastic>>(
          assembly, mpi_buffers, istep);
  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, elastic_psv>>(
          assembly, mpi_buffers, istep);
  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, elastic_sh>>(
          assembly, mpi_buffers, istep);
  dofs_updated += time_scheme.apply_corrector_phase_adjoint(elastic);
  dofs_updated += time_scheme.apply_corrector_phase_adjoint(elastic_psv);
  dofs_updated += time_scheme.apply_corrector_phase_adjoint(elastic_sh);

  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, poroelastic>>(
          assembly, mpi_buffers, istep);
  dofs_updated += time_scheme.apply_corrector_phase_adjoint(poroelastic);

  check_medium_step_update(assembly, dofs_updated, elements_updated, "adjoint");

  return { dofs_updated, elements_updated };
}

/**
 * @brief Advance the backward-reconstructed wavefield through one
 * predictor/corrector step across all media.
 *
 * Used by the @c combined solver to reverse-integrate the forward wavefield.
 * The coupling order is reversed relative to the forward/adjoint step (elastic
 * media are updated before acoustic), matching the reverse-time direction of
 * the reconstruction. Excludes @c elastic_psv_t, which the combined solver
 * does not support.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @tparam FieldType Wavefield tag to update (backward)
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @param time_scheme Time integration scheme driving the predictor/corrector
 * phases
 * @param assembly The assembly object containing the mesh and fields
 * @param mpi_buffers Solver-owned MPI buffers
 * @param istep Current time step
 * @return {degrees of freedom updated, elements updated}
 */
template <int NGLL, specfem::simulation::field_type FieldType,
          specfem::element::dimension_tag DimensionTag>
std::pair<int, int>
apply_backward_step(specfem::time_scheme::time_scheme &time_scheme,
                    specfem::assembly::assembly<DimensionTag> &assembly,
                    specfem::solver::MPIBuffers<DimensionTag> &mpi_buffers,
                    const int istep) {

  int dofs_updated = 0;
  int elements_updated = 0;

  dofs_updated += time_scheme.apply_predictor_phase_backward(elastic);
  dofs_updated += time_scheme.apply_predictor_phase_backward(elastic_psv);
  dofs_updated += time_scheme.apply_predictor_phase_backward(elastic_sh);
  dofs_updated += time_scheme.apply_predictor_phase_backward(acoustic);
  dofs_updated += time_scheme.apply_predictor_phase_backward(poroelastic);

  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, elastic>>(
          assembly, mpi_buffers, istep);
  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, elastic_psv>>(
          assembly, mpi_buffers, istep);
  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, elastic_sh>>(
          assembly, mpi_buffers, istep);
  dofs_updated += time_scheme.apply_corrector_phase_backward(elastic);
  dofs_updated += time_scheme.apply_corrector_phase_backward(elastic_psv);
  dofs_updated += time_scheme.apply_corrector_phase_backward(elastic_sh);

  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, acoustic>>(
          assembly, mpi_buffers, istep);
  dofs_updated += time_scheme.apply_corrector_phase_backward(acoustic);

  elements_updated +=
      update_medium<NGLL,
                    specfem::tags::Tags<DimensionTag, FieldType, poroelastic>>(
          assembly, mpi_buffers, istep);
  dofs_updated += time_scheme.apply_corrector_phase_backward(poroelastic);

  check_medium_step_update(assembly, dofs_updated, elements_updated,
                           "backward");

  return { dofs_updated, elements_updated };
}

} // namespace specfem::solver::impl
