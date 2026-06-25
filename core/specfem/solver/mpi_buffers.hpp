#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/element.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/simulation.hpp"
#include "specfem/utilities/errors.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::solver {

/**
 * @brief Solver-owned MPI buffers for field exchange during time-stepping.
 *
 * Pre-allocates, per (wavefield, medium), a `MediumBuffers` collection holding
 * one communication buffer per exchanged data class (acceleration and
 * mass_matrix). Each buffer exposes split-phase `begin_communicate` /
 * `finish_communicate` methods for overlapping halo exchange with computation,
 * and `reset` to free its device allocations once it is no longer needed.
 * Ownership lives in the solver so that the communication schedule can vary per
 * solver. To exchange a new data class, add a `CommBuffer` member to
 * `MediumBuffers` and a branch to its `get<DataClass>()`.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3).
 */
template <specfem::element::dimension_tag DimensionTag> struct MPIBuffers;

// ── dim3 helpers ───────────────────────────────────────────────────────────

namespace mpi_buffers_impl {

/**
 * @brief Split-phase wrapper around a single `mpi_buffer`.
 *
 * `begin_communicate` packs the field, fences, and posts the non-blocking
 * receive/send. `finish_communicate` waits for completion and accumulates the
 * received contributions back into the field. `reset` frees the device buffers.
 * When `active` is false (no buffer was created, or it was reset) all methods
 * are no-ops.
 *
 * @tparam FieldType Wavefield type (forward, backward, adjoint).
 * @tparam MediumTag Medium (elastic, acoustic).
 * @tparam DataClass Data class exchanged (acceleration or mass_matrix).
 */
template <specfem::simulation::field_type FieldType,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass>
struct CommBuffer {
  specfem::assembly::mpi_buffer<
      FieldType, specfem::element::dimension_tag::dim3, MediumTag, DataClass>
      buffer;
  bool active = false; ///< Whether a live buffer exists for this slot

  CommBuffer() = default;

  /**
   * @brief Phase 1: pack, fence, post receives, and send.
   *
   * @param field The simulation field containing the data to exchange.
   */
  void
  begin_communicate(const specfem::assembly::simulation_field<
                    specfem::element::dimension_tag::dim3, FieldType> &field) {
    if (!active)
      return;
    buffer.pack(field);
    Kokkos::fence("specfem::solver::CommBuffer::begin_communicate::pack");
    buffer.receive();
    buffer.send();
  }

  /**
   * @brief Phase 2: wait for MPI completion and unpack received contributions.
   *
   * @param field The simulation field to accumulate into.
   */
  void finish_communicate(
      specfem::assembly::simulation_field<specfem::element::dimension_tag::dim3,
                                          FieldType> &field) {
    if (!active)
      return;
    buffer.wait();
    buffer.unpack(field);
  }

  /// Free the per-neighbor device buffers and mark the slot inactive.
  void reset() {
    buffer.buffers.clear();
    active = false;
  }
};

/**
 * @brief Per-(wavefield, medium) collection of `CommBuffer`s, one per exchanged
 * data class.
 *
 * Holds one named `CommBuffer` per exchanged data class (acceleration and
 * mass_matrix). `get<DataClass>()` returns the buffer for a data class; to
 * exchange a new data class, add a `CommBuffer` member and a matching branch to
 * `get<DataClass>()`.
 *
 * @tparam TagsType Tags<wavefield_tag, medium_tag> identifying the slot.
 */
template <typename TagsType> struct MediumBuffers {
  /// Acceleration buffer, exchanged every time step.
  CommBuffer<TagsType::wavefield_tag, TagsType::medium_tag,
             specfem::data_access::DataClassType::acceleration>
      acceleration_buffer;
  /// Mass-matrix buffer, exchanged once during assembly then freed via `reset`.
  CommBuffer<TagsType::wavefield_tag, TagsType::medium_tag,
             specfem::data_access::DataClassType::mass_matrix>
      mass_matrix_buffer;

  /**
   * @brief Access the `CommBuffer` for one data class.
   *
   * @tparam DataClass Data class to access (acceleration or mass_matrix).
   */
  template <specfem::data_access::DataClassType DataClass> auto &get() {
    if constexpr (DataClass ==
                  specfem::data_access::DataClassType::acceleration)
      return acceleration_buffer;
    else if constexpr (DataClass ==
                       specfem::data_access::DataClassType::mass_matrix)
      return mass_matrix_buffer;
    else
      static_assert(
          specfem::utilities::always_false<DataClass>,
          "MediumBuffers only holds acceleration and mass_matrix buffers");
  }

  /// Free the buffer for one data class.
  template <specfem::data_access::DataClassType DataClass> void reset() {
    this->template get<DataClass>().reset();
  }
};

/// No-op single-buffer stand-in: accepts any field, does nothing.
struct NoopCommBuffer {
  template <class Field> void begin_communicate(const Field &) {}
  template <class Field> void finish_communicate(Field &) {}
  void reset() {}
};

/**
 * @brief No-op buffer collection for media without cross-rank exchange and for
 * the dim2 container.
 *
 * Mirrors `MediumBuffers`'s `get<DataClass>()` / `reset<DataClass>()` interface
 * so callers dispatch the same communication sequence over every medium
 * uniformly.
 */
struct NoopMediumBuffers {
  template <specfem::data_access::DataClassType> auto &get() {
    static NoopCommBuffer noop{};
    return noop;
  }
  template <specfem::data_access::DataClassType> void reset() {}
};

} // namespace mpi_buffers_impl

// ── dim2: no-op specialization ─────────────────────────────────────────────

template <> struct MPIBuffers<specfem::element::dimension_tag::dim2> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;

  MPIBuffers() = default;

  /// dim2 has no MPI exchange; every slot is a no-op.
  template <typename TagsType> auto &get() {
    static mpi_buffers_impl::NoopMediumBuffers slot{};
    return slot;
  }

  /// No-op: dim2 holds no device buffers to free.
  template <specfem::data_access::DataClassType> void reset() {}
};

// ── dim3: TypedStorage of unified buffers ──────────────────────────────────

template <> struct MPIBuffers<specfem::element::dimension_tag::dim3> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  // The set of (wavefield, medium) combinations that have cross-rank buffers
  static constexpr auto buffer_combos =
      WAVEFIELD_SET(forward, backward, adjoint) *
      specfem::assembly::mpi<dimension_tag>::media;

  /// Per-(wavefield, medium) unified buffer pair.
  template <typename TagsType>
  using buffers_for_tags = mpi_buffers_impl::MediumBuffers<TagsType>;

  /// Storage of unified buffers keyed by Tags<wavefield, medium>.
  specfem::tag_dispatch::TypedStorage<buffers_for_tags, decltype(buffer_combos)>
      buffers;

  /// True when the given medium participates in cross-rank exchange. Derived
  /// from the assembly's mpi media so the set is stated in exactly one place.
  static constexpr bool has_buffer(const specfem::element::medium_tag medium) {
    for (const auto exchanging_medium :
         specfem::assembly::mpi<dimension_tag>::media.values) {
      if (exchanging_medium == medium)
        return true;
    }
    return false;
  }

  MPIBuffers() = default;

  /**
   * @brief Construct pre-allocated buffers from MPI interfaces.
   *
   * Only allocates buffers for the field types relevant to the simulation mode,
   * read from @p mpi_obj: forward-only creates forward buffers; combined
   * creates backward and adjoint buffers. Each active slot receives one buffer
   * per exchanged data class. Unused slots remain default-constructed
   * (inactive).
   *
   * @param mpi_obj Fully-constructed MPI communication object; its `simulation`
   * member selects which field types are allocated.
   */
  MPIBuffers(const specfem::assembly::mpi<dimension_tag> &mpi_obj)
      : buffers([&]<typename TagsType>() -> buffers_for_tags<TagsType> {
          constexpr auto acceleration =
              specfem::data_access::DataClassType::acceleration;
          constexpr auto mass_matrix =
              specfem::data_access::DataClassType::mass_matrix;
          const bool should_create =
              (TagsType::wavefield_tag ==
                   specfem::simulation::field_type::forward &&
               mpi_obj.simulation == specfem::simulation::type::forward) ||
              (TagsType::wavefield_tag !=
                   specfem::simulation::field_type::forward &&
               mpi_obj.simulation == specfem::simulation::type::combined);
          buffers_for_tags<TagsType> slot{};
          if (should_create) {
            slot.template get<acceleration>()
                .buffer = mpi_obj.template create_mpi_buffer<
                TagsType::wavefield_tag, TagsType::medium_tag, acceleration>();
            slot.template get<acceleration>().active = true;
            slot.template get<mass_matrix>()
                .buffer = mpi_obj.template create_mpi_buffer<
                TagsType::wavefield_tag, TagsType::medium_tag, mass_matrix>();
            slot.template get<mass_matrix>().active = true;
          }
          return slot;
        }) {}

  /**
   * @brief Access the buffer collection for a (wavefield, medium) slot.
   *
   * @tparam TagsType Tags<wavefield_tag, medium_tag, ...> identifying the slot.
   */
  template <typename TagsType> auto &get() {
    if constexpr (has_buffer(TagsType::medium_tag)) {
      using key =
          specfem::tags::Tags<TagsType::wavefield_tag, TagsType::medium_tag>;
      return buffers.template get<key>();
    } else {
      static mpi_buffers_impl::NoopMediumBuffers noop{};
      return noop;
    }
  }

  /**
   * @brief Free the given data class buffer across all (wavefield, medium)
   * slots.
   *
   * @tparam DataClass Data class to free (acceleration or mass_matrix).
   */
  template <specfem::data_access::DataClassType DataClass> void reset() {
    specfem::tag_dispatch::for_each(buffer_combos, [&]<typename TagsType>() {
      buffers.template get<TagsType>().template reset<DataClass>();
    });
  }
};

/**
 * @brief Build solver-owned MPI buffers from an assembly.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3).
 * @param assembly The assembly object providing the MPI interfaces (dim3); the
 * simulation mode is read from its `mpi_interfaces`.
 * @return A populated `MPIBuffers<DimensionTag>`.
 */
template <specfem::element::dimension_tag DimensionTag>
MPIBuffers<DimensionTag>
make_mpi_buffers(const specfem::assembly::assembly<DimensionTag> &assembly) {
  if constexpr (DimensionTag == specfem::element::dimension_tag::dim3) {
    return MPIBuffers<DimensionTag>(assembly.mpi_interfaces);
  } else {
    (void)assembly;
    return MPIBuffers<DimensionTag>{};
  }
}

} // namespace specfem::solver
