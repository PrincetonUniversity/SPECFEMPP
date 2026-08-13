#pragma once

#include "specfem/assembly/fields/fields.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>

namespace specfem {
namespace solver {

/**
 * @brief In-memory ring buffer holding @p window_size forward displacement
 *        snapshots for one UNDO_ATTENUATION subset window.
 *
 * Equivalent to Fortran's @c b_displ_elastic_buffer(NDIM, NGLOB_AB,
 * NT_DUMP_ATTENUATION). Only the displacement field is buffered — velocity and
 * acceleration of the reconstructed forward wavefield are not needed for the
 * kernel formulas (strain is recomputed on-the-fly from the displacement
 * gradient). This matches the Fortran comment:
 *
 *   "only the displacement needs to be stored in memory buffers in order to
 *    compute the sensitivity kernels, not the memory variables R_ij, because
 *    the sensitivity kernel calculations only involve the displacement and the
 *    strain, and the strain can be recomputed on the fly"
 *
 * Usage (one subset window):
 *
 *   // SUB-PHASE A: forward replay
 *   for (int k = 0; k < window_size; ++k) {
 *     advance_forward_one_step(assembly);
 *     buffer.store(assembly.fields.forward, k);
 *   }
 *
 *   // SUB-PHASE B: adjoint pass (reversed)
 *   for (int k = window_size - 1; k >= 0; --k) {
 *     buffer.load(assembly.fields.forward, k);
 *     advance_adjoint_one_step(assembly);
 *     compute_derivatives(assembly);
 *   }
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
struct ForwardDisplacementBuffer {

  /// Device view holding all displacement snapshots for one medium.
  /// Shape: [window_size, nglob_medium, components_per_medium]
  using BufferView = Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                  Kokkos::DefaultExecutionSpace>;

  using forward_field_type = specfem::assembly::simulation_field<
      DimensionTag, specfem::simulation::field_type::forward>;

  /// Per-medium buffer stored as a homogeneous Storage<BufferView, ET>.
  specfem::tag_dispatch::Storage<BufferView,
                                 decltype(forward_field_type::combinations)>
      storage;

  int window_size_ = 0;

  ForwardDisplacementBuffer() = default;

  /**
   * @brief Allocate the buffer for one subset window.
   *
   * @param window_size  Number of forward steps to buffer per subset
   *                     (= NT_DUMP_ATTENUATION for fixed-stride checkpointing)
   * @param fwd          Forward simulation field (provides nglob per medium)
   */
  void allocate(int window_size, const forward_field_type &fwd) {
    window_size_ = window_size;
    specfem::tag_dispatch::for_each(
        forward_field_type::combinations, [&]<typename TagsType>() {
          constexpr auto medium_tag = TagsType::medium_tag;
          const int nglob = fwd.template get_nglob<medium_tag>();
          if (nglob > 0) {
            constexpr int components =
                specfem::element::attributes<DimensionTag,
                                             medium_tag>::components;
            storage.template get<TagsType>() = BufferView(
                "fwd_displ_buf_" + specfem::element::to_string(medium_tag),
                window_size, nglob, components);
          }
        });
  }

  /**
   * @brief Store the displacement of @p fwd into slot @p slot.
   *
   * @param fwd   Forward simulation field (device-side displacement is read)
   * @param slot  Slot index in [0, window_size)
   */
  void store(const forward_field_type &fwd, int slot) const {
    specfem::tag_dispatch::for_each(
        forward_field_type::combinations, [&]<typename TagsType>() {
          constexpr auto medium_tag = TagsType::medium_tag;
          const int nglob = fwd.template get_nglob<medium_tag>();
          if (nglob > 0) {
            const auto &buf = storage.template get<TagsType>();
            auto slot_view =
                Kokkos::subview(buf, slot, Kokkos::ALL, Kokkos::ALL);
            Kokkos::deep_copy(slot_view,
                              fwd.template get_field<medium_tag>().get_field());
          }
        });
  }

  /**
   * @brief Restore displacement from slot @p slot into @p fwd.
   *
   * Only the displacement component is overwritten; velocity and acceleration
   * remain unchanged.
   *
   * @param fwd   Forward simulation field (displacement is overwritten)
   * @param slot  Slot index in [0, window_size)
   */
  void load(forward_field_type &fwd, int slot) const {
    specfem::tag_dispatch::for_each(
        forward_field_type::combinations, [&]<typename TagsType>() {
          constexpr auto medium_tag = TagsType::medium_tag;
          const int nglob = fwd.template get_nglob<medium_tag>();
          if (nglob > 0) {
            const auto &buf = storage.template get<TagsType>();
            auto slot_view =
                Kokkos::subview(buf, slot, Kokkos::ALL, Kokkos::ALL);
            Kokkos::deep_copy(fwd.template get_field<medium_tag>().get_field(),
                              slot_view);
          }
        });
  }

  /// @brief Returns the configured window size.
  int window_size() const { return window_size_; }
};

} // namespace solver
} // namespace specfem
