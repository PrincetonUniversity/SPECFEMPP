#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/element.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/simulation.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <tuple>
#include <utility>

namespace specfem::solver {

/**
 * @brief Solver-owned MPI buffers for field exchange during time-stepping.
 *
 * Pre-allocates, per (wavefield, medium), a `MediumBuffers` collection holding
 * one communication buffer per data class in `exchanged_data_classes`
 * (currently acceleration and mass_matrix). Each buffer exposes split-phase
 * `begin_communicate` / `finish_communicate` methods for overlapping halo
 * exchange with computation, and `reset` to free its device allocations once it
 * is no longer needed. Ownership lives in the solver so that the communication
 * schedule can vary per solver. The set of exchanged data classes is
 * data-driven: adding one to `exchanged_data_classes` extends every (wavefield,
 * medium) slot.
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
 * @brief Compile-time list of data classes exchanged over MPI.
 *
 * A non-type-template-parameter pack carrier — purely a type, with no runtime
 * storage. The pack is unpacked at compile time to build the per-medium buffer
 * tuple, to look up a data class's slot index, and to iterate data classes
 * during construction.
 *
 * @tparam DataClasses The data classes carried by this list.
 */
template <specfem::data_access::DataClassType... DataClasses>
struct data_class_list {};

/**
 * @brief The data classes a medium exchanges across ranks.
 *
 * Single edit point for the data-class axis: to exchange a new field (e.g.
 * displacement), add its `DataClassType` here and every consumer (per-medium
 * buffer tuple, `MediumBuffers::get` / `reset`, the construction loop) picks it
 * up automatically.
 */
using exchanged_data_classes =
    data_class_list<specfem::data_access::DataClassType::acceleration,
                    specfem::data_access::DataClassType::mass_matrix>;

/**
 * @brief Compile-time position of @p Target within a `data_class_list`.
 *
 * @tparam Target      The data class to locate.
 * @tparam DataClasses The data classes carried by the list (deduced).
 * @param  list        A `data_class_list` instance (its pack is deduced).
 * @return Zero-based index of @p Target within the list.
 */
template <specfem::data_access::DataClassType Target,
          specfem::data_access::DataClassType... DataClasses>
constexpr std::size_t index_of(data_class_list<DataClasses...> list) {
  (void)list;
  static_assert(((Target == DataClasses) || ...),
                "DataClass is not a registered exchanged data class");
  std::size_t index = 0;
  ((Target == DataClasses ? true : (++index, false)) || ...);
  return index;
}

/**
 * @brief Invoke @p func once per data class in @p list (compile-time fold).
 *
 * @p func must be a generic callable taking a `DataClassType` non-type template
 * parameter, i.e. `[]<specfem::data_access::DataClassType DataClass>() { ...
 * }`.
 *
 * @tparam DataClasses The data classes carried by the list (deduced).
 * @tparam Func        The per-data-class callable.
 * @param  list        A `data_class_list` instance (its pack is deduced).
 * @param  func        The callable invoked once per data class.
 */
template <specfem::data_access::DataClassType... DataClasses, typename Func>
void for_each_data_class(data_class_list<DataClasses...> list, Func &&func) {
  (void)list;
  (func.template operator()<DataClasses>(), ...);
}

/// Convenience overload iterating over `exchanged_data_classes`.
template <typename Func> void for_each_data_class(Func &&func) {
  for_each_data_class(exchanged_data_classes{}, std::forward<Func>(func));
}

/**
 * @brief Primary template for the per-(wavefield, medium) buffer tuple.
 *
 * @tparam FieldType     Wavefield type (forward, backward, adjoint).
 * @tparam MediumTag     Medium (elastic, acoustic).
 * @tparam DataClassList The `data_class_list` to expand.
 */
template <specfem::simulation::field_type FieldType,
          specfem::element::medium_tag MediumTag, typename DataClassList>
struct medium_buffer_tuple;

/// Partial specialization expanding the `data_class_list` pack into one
/// `CommBuffer` per data class.
template <specfem::simulation::field_type FieldType,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType... DataClasses>
struct medium_buffer_tuple<FieldType, MediumTag,
                           data_class_list<DataClasses...>> {
  using type = std::tuple<CommBuffer<FieldType, MediumTag, DataClasses>...>;
};

/// Alias for `medium_buffer_tuple<...>::type`.
template <specfem::simulation::field_type FieldType,
          specfem::element::medium_tag MediumTag, typename DataClassList>
using medium_buffer_tuple_t =
    typename medium_buffer_tuple<FieldType, MediumTag, DataClassList>::type;

/**
 * @brief Per-(wavefield, medium) collection of `CommBuffer`s, one per exchanged
 * data class.
 *
 * Holds a heterogeneous tuple of `CommBuffer`s keyed by position in
 * `exchanged_data_classes`. `get<DataClass>()` returns the buffer for a data
 * class; adding a data class to `exchanged_data_classes` grows this collection
 * automatically.
 *
 * @tparam TagsType Tags<wavefield_tag, medium_tag> identifying the slot.
 */
template <typename TagsType> struct MediumBuffers {
  /// One `CommBuffer` per data class in `exchanged_data_classes`.
  medium_buffer_tuple_t<TagsType::wavefield_tag, TagsType::medium_tag,
                        exchanged_data_classes>
      buffers;

  /**
   * @brief Access the `CommBuffer` for one data class.
   *
   * @tparam DataClass Data class to access; must be in
   * `exchanged_data_classes`.
   */
  template <specfem::data_access::DataClassType DataClass> auto &get() {
    constexpr std::size_t index =
        specfem::solver::mpi_buffers_impl::index_of<DataClass>(
            exchanged_data_classes{});
    return std::get<index>(buffers);
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

  /// All (wavefield, medium) combinations with cross-rank exchange.
  static constexpr auto buffer_combos =
      WAVEFIELD_SET(forward, backward, adjoint) * MEDIUM_SET(elastic, acoustic);

  /// Per-(wavefield, medium) unified buffer pair.
  template <typename TagsType>
  using buffers_for_tags = mpi_buffers_impl::MediumBuffers<TagsType>;

  /// Storage of unified buffers keyed by Tags<wavefield, medium>.
  specfem::tag_dispatch::TypedStorage<buffers_for_tags, decltype(buffer_combos)>
      buffers;

  /// True when the given medium has a cross-rank exchange buffer.
  static constexpr bool has_buffer(const specfem::element::medium_tag medium) {
    return medium == specfem::element::medium_tag::elastic ||
           medium == specfem::element::medium_tag::acoustic;
  }

  MPIBuffers() = default;

  /**
   * @brief Construct pre-allocated buffers from MPI interfaces.
   *
   * Only allocates buffers for the field types relevant to the simulation mode:
   * forward-only creates forward buffers; combined creates backward and adjoint
   * buffers. Each active slot receives one buffer per data class in
   * `exchanged_data_classes`. Unused slots remain default-constructed
   * (inactive).
   *
   * @param mpi_obj    Fully-constructed MPI communication object.
   * @param simulation Simulation mode (forward, combined, etc.).
   */
  MPIBuffers(const specfem::assembly::mpi<dimension_tag> &mpi_obj,
             const specfem::simulation::type simulation)
      : buffers([&]<typename TagsType>() -> buffers_for_tags<TagsType> {
          const bool should_create =
              (TagsType::wavefield_tag ==
                   specfem::simulation::field_type::forward &&
               simulation == specfem::simulation::type::forward) ||
              (TagsType::wavefield_tag !=
                   specfem::simulation::field_type::forward &&
               simulation == specfem::simulation::type::combined);
          buffers_for_tags<TagsType> slot{};
          if (should_create) {
            mpi_buffers_impl::for_each_data_class(
                [&]<specfem::data_access::DataClassType DataClass>() {
                  slot.template get<DataClass>().buffer =
                      mpi_obj.template create_mpi_buffer<
                          TagsType::wavefield_tag, TagsType::medium_tag,
                          DataClass>();
                  slot.template get<DataClass>().active = true;
                });
          }
          return slot;
        }) {}

  /**
   * @brief Access the buffer collection for a (wavefield, medium) slot.
   *
   * Returns the real buffer collection for media with cross-rank exchange
   * (elastic, acoustic); for any other medium it returns a shared no-op
   * collection, so callers can dispatch the same communication sequence over
   * every medium uniformly.
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
   * Used to deallocate the mass-matrix buffers once the mass matrix has been
   * assembled and inverted, since they are not needed during time-stepping.
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
 * dim3 buffers are constructed from `assembly.mpi_interfaces`; dim2 has no MPI
 * exchange and yields a default (no-op) container.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3).
 * @param assembly The assembly object providing the MPI interfaces (dim3).
 * @param simulation Simulation mode (forward, combined, etc.).
 * @return A populated `MPIBuffers<DimensionTag>`.
 */
template <specfem::element::dimension_tag DimensionTag>
MPIBuffers<DimensionTag>
make_mpi_buffers(const specfem::assembly::assembly<DimensionTag> &assembly,
                 const specfem::simulation::type simulation) {
  if constexpr (DimensionTag == specfem::element::dimension_tag::dim3) {
    return MPIBuffers<DimensionTag>(assembly.mpi_interfaces, simulation);
  } else {
    (void)assembly;
    (void)simulation;
    return MPIBuffers<DimensionTag>{};
  }
}

} // namespace specfem::solver
