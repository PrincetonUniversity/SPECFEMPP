#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace specfem {
namespace linear_system {

/**
 * @brief What one index slot of a dof selection may be.
 *
 * Three kinds, which together let a selection name one dof, one element, or an
 * arbitrary set of either:
 *
 * - an integral: that single index;
 * - `Kokkos::ALL`: the whole extent of the slot;
 * - an index container: the listed indices.
 *
 * "Index container" is stated structurally rather than by naming types, so that
 * `Kokkos::View<int *>` and specfem::datatype::ElementIndexRange -- which is
 * what specfem::linear_system::Mapping::elements already returns -- both
 * qualify without conversion.
 *
 * The container requirements are written on the *decayed* result types: a
 * `Kokkos::View` subscript yields `int &`, and a reference is not itself an
 * integral type.
 */
template <typename T>
concept IndexSelector =
    std::integral<T> || std::is_same_v<std::remove_cvref_t<T>, Kokkos::ALL_t> ||
    requires(const T &t, int i) {
      requires std::integral<std::remove_cvref_t<decltype(t(i))>>;
      requires std::integral<std::remove_cvref_t<decltype(t.extent(0))>>;
    };

} // namespace linear_system

namespace linear_system_impl {

/**
 * @brief Number of indices a selector names.
 *
 * @tparam Selector Slot value; see specfem::linear_system::IndexSelector
 * @param selector Slot value
 * @param full_extent Extent of the slot, used only by `Kokkos::ALL`
 * @return Count of indices the slot expands to
 */
template <typename Selector>
constexpr int selector_extent(const Selector &selector, const int full_extent) {
  if constexpr (std::integral<Selector>) {
    return 1;
  } else if constexpr (std::is_same_v<std::remove_cvref_t<Selector>,
                                      Kokkos::ALL_t>) {
    return full_extent;
  } else {
    return static_cast<int>(selector.extent(0));
  }
}

/**
 * @brief The `k`-th index a selector names.
 *
 * @tparam Selector Slot value; see specfem::linear_system::IndexSelector
 * @param selector Slot value
 * @param k Offset in `[0, selector_extent(selector, ...))`
 * @return Index at that offset
 */
template <typename Selector>
constexpr int selector_at(const Selector &selector, const int k) {
  if constexpr (std::integral<Selector>) {
    return static_cast<int>(selector);
  } else if constexpr (std::is_same_v<std::remove_cvref_t<Selector>,
                                      Kokkos::ALL_t>) {
    return k;
  } else {
    return static_cast<int>(selector(k));
  }
}

} // namespace linear_system_impl

namespace linear_system {

/**
 * @brief An ordered set of global dof ids, named in mesh coordinates.
 *
 * Produced by specfem::linear_system::Mapping::operator() when any index slot
 * is a non-integral selector, and consumed by
 * specfem::linear_system::SparseMatrixView as the row or column set of a matrix
 * update:
 *
 * @code
 * const auto edof = mapping(ispec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
 *                           Kokkos::ALL);
 * K(edof, edof) += k_e;
 * @endcode
 *
 * Nothing is allocated or expanded at construction: the set holds its selectors
 * and decodes an id on demand, so naming the dofs of an element costs no more
 * than the loop that consumes them.
 *
 * ### Ordering
 *
 * The sequence is ordered, slowest-varying index first:
 *
 * ```
 * ispec -> icomp -> iz -> iy -> ix     (five slots)
 * iglob -> icomp                       (two slots)
 * ```
 *
 * Two consequences are worth stating outright, because neither follows from
 * reading the argument list left to right:
 *
 * - `icomp` varies *more slowly* than `iz`, `iy` and `ix`, so the written slot
 *   order `(ispec, iz, iy, ix, icomp)` deliberately does not imply C-order.
 *   This is forced by @ref local_dof_index, which defines the row and column
 *   ordering of the element stiffness blocks; with a scalar `ispec` the
 *   sequence reduces to `local_dof_index` exactly.
 * - `ispec` is the slowest of all, so a multi-element set is the concatenation
 *   of the per-element sets rather than an interleaving of them. That is what
 *   makes @ref expand_block well defined, and what lets one dense block per
 *   element be scattered from a single rank-3 update.
 *
 * ### Lifetime
 *
 * The mapping is borrowed, not owned: a `DofSet` must not outlive it. The
 * reference member also deletes copy-assignment, so the intended loop-local use
 * is what the type permits.
 *
 * @tparam MappingType Dof numbering the ids are drawn from; see
 *                     specfem::linear_system::Mapping
 * @tparam Selectors Slot values; two or five, each an @ref IndexSelector
 */
template <typename MappingType, typename... Selectors>
  requires((IndexSelector<Selectors> && ...) &&
           (sizeof...(Selectors) == 2 || sizeof...(Selectors) == 5))
class DofSet {
public:
  /// Integer type of the global dof ids
  using global_ordinal_type = typename MappingType::global_ordinal_type;

  /**
   * @brief Bind selectors to a mapping.
   *
   * Prefer specfem::linear_system::Mapping::operator() over constructing this
   * directly -- it normalizes a `Kokkos::ALL` element slot to the medium's
   * element list, which this constructor assumes has already happened.
   *
   * @param mapping Dof numbering to draw ids from; must outlive the set
   * @param selectors Slot values
   */
  constexpr DofSet(const MappingType &mapping, Selectors... selectors)
      : mapping_(mapping), selectors_(selectors...) {}

  /// Number of dof ids in the set
  constexpr int size() const { return outer_extent() * inner_size(); }

  /**
   * @brief Number of indices the leading slot names.
   *
   * The element count of a multi-element set, or the point count of a
   * multi-point one; 1 when the leading slot is a scalar.
   */
  constexpr int outer_extent() const {
    return specfem::linear_system_impl::selector_extent(std::get<0>(selectors_),
                                                        leading_slot_extent());
  }

  /// Number of dof ids contributed by each leading index; @ref size divided by
  /// @ref outer_extent
  constexpr int inner_size() const {
    if constexpr (num_slots == 5) {
      return extent_of<4>() * extent_of<1>() * extent_of<2>() * extent_of<3>();
    } else {
      return extent_of<1>();
    }
  }

  /**
   * @brief The `k`-th global dof id, in the order documented above.
   *
   * @param k Offset in `[0, size())`
   * @return Global dof id
   */
  constexpr global_ordinal_type operator[](const int k) const {
    if constexpr (num_slots == 5) {
      const int nz = extent_of<1>();
      const int ny = extent_of<2>();
      const int nx = extent_of<3>();

      const int inner = inner_size();
      const int e = k / inner;
      int rest = k % inner;

      // icomp is the slowest of the trailing four; see the class docs.
      const int icomp = rest / (nz * ny * nx);
      rest %= nz * ny * nx;
      const int iz = rest / (ny * nx);
      rest %= ny * nx;
      const int iy = rest / nx;
      const int ix = rest % nx;

      return mapping_(at_of<0>(e), at_of<1>(iz), at_of<2>(iy), at_of<3>(ix),
                      at_of<4>(icomp));
    } else {
      const int ncomp = extent_of<1>();
      return mapping_(at_of<0>(k / ncomp), at_of<1>(k % ncomp));
    }
  }

  /**
   * @brief Write every id of the set, in order.
   *
   * @tparam OutputIt Output iterator over @ref global_ordinal_type
   * @param out Destination; @ref size ids are written
   */
  template <typename OutputIt> constexpr void expand(OutputIt out) const {
    const int n = size();
    for (int k = 0; k < n; ++k) {
      *out++ = (*this)[k];
    }
  }

  /**
   * @brief Write the ids contributed by one leading index.
   *
   * `expand_block(e, out)` writes the same @ref inner_size ids, in the same
   * order, that @ref expand would write for a set whose leading slot were the
   * scalar `e`-th index.
   *
   * @tparam OutputIt Output iterator over @ref global_ordinal_type
   * @param e Offset in `[0, outer_extent())`
   * @param out Destination; @ref inner_size ids are written
   */
  template <typename OutputIt>
  constexpr void expand_block(const int e, OutputIt out) const {
    const int inner = inner_size();
    const int base = e * inner;
    for (int j = 0; j < inner; ++j) {
      *out++ = (*this)[base + j];
    }
  }

private:
  /// Number of index slots: 5 for `(ispec, iz, iy, ix, icomp)`, 2 for
  /// `(iglob, icomp)`
  constexpr static std::size_t num_slots = sizeof...(Selectors);

  /// Full extent of slot `Slot`, used when that slot is `Kokkos::ALL`
  template <std::size_t Slot> constexpr int slot_extent() const {
    if constexpr (num_slots == 5) {
      if constexpr (Slot == 0) {
        return mapping_.nelements();
      } else if constexpr (Slot == 1) {
        return mapping_.ngllz();
      } else if constexpr (Slot == 2) {
        return mapping_.nglly();
      } else if constexpr (Slot == 3) {
        return mapping_.ngllx();
      } else {
        return mapping_.ncomp();
      }
    } else {
      if constexpr (Slot == 0) {
        return mapping_.nglob();
      } else {
        return mapping_.ncomp();
      }
    }
  }

  /// @ref slot_extent of the leading slot
  constexpr int leading_slot_extent() const { return slot_extent<0>(); }

  /// Number of indices slot `Slot` names
  template <std::size_t Slot> constexpr int extent_of() const {
    return specfem::linear_system_impl::selector_extent(
        std::get<Slot>(selectors_), slot_extent<Slot>());
  }

  /// The `k`-th index slot `Slot` names
  template <std::size_t Slot> constexpr int at_of(const int k) const {
    return specfem::linear_system_impl::selector_at(std::get<Slot>(selectors_),
                                                    k);
  }

  const MappingType &mapping_; ///< Borrowed; the set must not outlive it
  std::tuple<Selectors...> selectors_; ///< Slot values
};

} // namespace linear_system
} // namespace specfem
