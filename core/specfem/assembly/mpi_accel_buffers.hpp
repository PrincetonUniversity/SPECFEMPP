#pragma once

#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/element.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/simulation.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Persistent MPI buffers for acceleration field exchange during
 * time-stepping.
 *
 * Pre-allocates per-medium `mpi_buffer<..., acceleration>` objects at
 * construction time, then provides split-phase `begin_exchange` /
 * `complete_exchange` methods for overlapping communication with computation.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3).
 */
template <specfem::element::dimension_tag DimensionTag>
struct mpi_accel_buffers;

// ── dim2: no-op specialization ─────────────────────────────────────────────

template <> struct mpi_accel_buffers<specfem::element::dimension_tag::dim2> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;

  mpi_accel_buffers() = default;

  template <specfem::simulation::field_type FieldType,
            specfem::element::medium_tag MediumTag>
  void begin_exchange(
      const specfem::assembly::simulation_field<dimension_tag, FieldType> &) {}

  template <specfem::simulation::field_type FieldType,
            specfem::element::medium_tag MediumTag>
  void complete_exchange(
      specfem::assembly::simulation_field<dimension_tag, FieldType> &) {}
};

// ── dim3: TypedStorage-based buffers ───────────────────────────────────────

template <> struct mpi_accel_buffers<specfem::element::dimension_tag::dim3> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  /// Acceleration data class type for MPI buffer template
  constexpr static auto accel_dct =
      specfem::data_access::DataClassType::acceleration;

  /// All (wavefield, medium) combinations for buffer storage
  static constexpr auto buffer_combos =
      WAVEFIELD_SET(forward, backward, adjoint) * MEDIUM_SET(elastic, acoustic);

  /// Maps a Tags<wavefield_tag, medium_tag> to the concrete mpi_buffer type
  template <typename TagsType>
  using buffer_for_tags =
      specfem::assembly::mpi_buffer<TagsType::wavefield_tag, dimension_tag,
                                    TagsType::medium_tag, accel_dct>;

  /// Check at compile time whether a (wavefield, medium) pair has a buffer
  template <specfem::simulation::field_type FT, specfem::element::medium_tag MT>
  static constexpr bool has_buffer() {
    using ET = std::remove_const_t<decltype(buffer_combos)>;
    for (const auto &c : ET::combos)
      if (c.template get<0>() == FT && c.template get<1>() == MT)
        return true;
    return false;
  }

  /// Per-(wavefield, medium) buffer storage
  specfem::tag_dispatch::TypedStorage<buffer_for_tags, decltype(buffer_combos)>
      buffers;

  mpi_accel_buffers() = default;

  /**
   * @brief Construct pre-allocated acceleration buffers from MPI interfaces.
   *
   * Only allocates buffers for the field types relevant to the simulation
   * mode: forward-only creates forward buffers; combined creates backward
   * and adjoint buffers. Unused slots remain default-constructed (empty).
   *
   * @param mpi_obj        Fully-constructed MPI communication object.
   * @param simulation     Simulation mode (forward, combined, etc.).
   */
  mpi_accel_buffers(const specfem::assembly::mpi<dimension_tag> &mpi_obj,
                    const specfem::simulation::type simulation)
      : buffers([&]<typename TagsType>() -> buffer_for_tags<TagsType> {
          constexpr auto ft = TagsType::wavefield_tag;
          const bool should_create =
              (ft == specfem::simulation::field_type::forward &&
               simulation == specfem::simulation::type::forward) ||
              (ft != specfem::simulation::field_type::forward &&
               simulation == specfem::simulation::type::combined);
          if (should_create)
            return mpi_obj.template create_mpi_buffer<ft, TagsType::medium_tag,
                                                      accel_dct>();
          return {};
        }) {}

  /**
   * @brief Phase 1: pack acceleration, fence, post receives, and send.
   *
   * Call after computing stiffness on outer elements, before computing
   * stiffness on inner elements. GPU fence ensures the pack kernel completes
   * before MPI sends begin.
   *
   * @tparam FieldType  Wavefield type (forward, backward, adjoint).
   * @tparam MediumTag  Medium (elastic, acoustic).
   * @param field       The simulation field containing acceleration data.
   */
  template <specfem::simulation::field_type FieldType,
            specfem::element::medium_tag MediumTag>
  void
  begin_exchange(const specfem::assembly::simulation_field<dimension_tag,
                                                           FieldType> &field) {
    if constexpr (has_buffer<FieldType, MediumTag>()) {
      using key = specfem::tags::Tags<FieldType, MediumTag>;
      auto &buf = buffers.template get<key>();
      buf.pack(field);
      Kokkos::fence("mpi_accel_buffers::begin_exchange::pack_fence");
      buf.receive();
      buf.send();
    }
  }

  /**
   * @brief Phase 2: wait for MPI completion and unpack received acceleration.
   *
   * Call after computing stiffness on inner elements. Blocks until all
   * MPI communication completes, then accumulates received acceleration
   * contributions into the field.
   *
   * @tparam FieldType  Wavefield type (forward, backward, adjoint).
   * @tparam MediumTag  Medium (elastic, acoustic).
   * @param field       The simulation field to accumulate into.
   */
  template <specfem::simulation::field_type FieldType,
            specfem::element::medium_tag MediumTag>
  void complete_exchange(
      specfem::assembly::simulation_field<dimension_tag, FieldType> &field) {
    if constexpr (has_buffer<FieldType, MediumTag>()) {
      using key = specfem::tags::Tags<FieldType, MediumTag>;
      auto &buf = buffers.template get<key>();
      buf.wait();
      buf.unpack(field);
    }
  }
};

} // namespace specfem::assembly
