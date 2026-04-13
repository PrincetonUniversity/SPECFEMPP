#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/tag_dispatch/element_combinations.hpp"
#include "specfem/tag_dispatch/storage.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>

namespace specfem::assembly::receivers_impl {

/**
 * @brief Base class for per-material receiver index stores.
 *
 * Replaces the @c FOR_EACH_IN_PRODUCT ... @c DECLARE / @c CAPTURE patterns
 * that previously generated per-combination index-view members at
 * compile-time.  Instead, a @ref specfem::tag_dispatch::Storage is used to
 * keep one index view per valid (dimension, medium, property, attenuation)
 * combination, mirroring the approach taken by
 * @ref specfem::assembly::element_types_impl::element_types_base.
 *
 * For each valid material combination two index views are maintained:
 *  - @c receiver_elements_   – spectral-element index of each receiver in the
 *                              group.
 *  - @c receiver_indices_    – global receiver index of each receiver in the
 *                              group.
 *
 * Both are populated by calling @ref build_index_stores once the
 * all-receivers @c h_elements view (receiver → element mapping) has been
 * filled.  @ref get_indices_on_device and @ref get_indices_on_host provide
 * runtime-tag lookup into those stores.
 *
 * @tparam ElementSets  Tag-set descriptor type, e.g.
 *         @c specfem::assembly::element_types_impl::ElementSets<DimTag>.
 *         Must expose @c dimension_tag, @c medium_set, @c property_set, and
 *         @c attenuation_set.
 */
template <typename ElementSets> struct ReceiverIndexBase {
  static constexpr auto dimension_tag = ElementSets::dimension_tag;

  /// All valid (dimension, medium, property, attenuation) combinations for
  /// this dimension.
  static constexpr auto combinations_by_material =
      specfem::tag_dispatch::dimension_set<dimension_tag>{} *
      ElementSets::medium_set * ElementSets::property_set *
      ElementSets::attenuation_set;

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using HostIndexViewType =
      Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>;

private:
  using MaterialCombos = decltype(combinations_by_material);

  template <typename Sets>
  using IndexStorage = specfem::tag_dispatch::Storage<IndexViewType, Sets>;

  template <typename Sets>
  using HostIndexStorage =
      specfem::tag_dispatch::Storage<HostIndexViewType, Sets>;

  IndexStorage<MaterialCombos> receiver_elements_;
  HostIndexStorage<MaterialCombos> h_receiver_elements_;
  IndexStorage<MaterialCombos> receiver_indices_;
  HostIndexStorage<MaterialCombos> h_receiver_indices_;

protected:
  /**
   * @brief Populate all per-material receiver index stores.
   *
   * Iterates over all valid material combinations via
   * @ref tag_dispatch::Storage 's initialiser-lambda interface (same pattern
   * as @c element_types_base) and, for each combination, builds host-side
   * index views that are then mirrored to the device.
   *
   * @param h_elements_all  Host view, size @p nrec, mapping receiver index →
   *                        spectral-element index.  A value of -1 is a
   *                        sentinel meaning the receiver was not located in
   *                        the current MPI partition; such entries are skipped.
   * @param element_types   Element-type classifier for the current mesh
   *                        partition.
   */
  void build_index_stores(
      const HostIndexViewType &h_elements_all,
      const specfem::assembly::element_types<dimension_tag> &element_types) {

    // ── host stores ────────────────────────────────────────────────────────
    // For each valid (dim, medium, property, attenuation) combination: count
    // matching receivers, allocate a view of that size, then fill it.
    //
    // Generic factory: returns a Storage initializer lambda that selects
    // receivers whose host element has tags matching the compile-time TagsType.
    // get_value(irec, ispec) supplies the integer stored for each match.
    // tag_views are callables (int ispec) -> tag, matched via TagsType{}.has().
    auto make_initializer = [&](std::string label_prefix, auto get_value,
                                auto... tag_views) {
      return [&, label_prefix, get_value,
              tag_views...]<typename TagsType>() -> HostIndexViewType {
        int count = 0;
        for (int irec = 0; irec < (int)h_elements_all.extent(0); ++irec) {
          const int ispec = h_elements_all(irec);
          if (ispec < 0)
            continue;
          if (TagsType{}.has(tag_views(ispec)...))
            ++count;
        }
        HostIndexViewType h_view(label_prefix + TagsType::name(), count);
        int idx = 0;
        for (int irec = 0; irec < (int)h_elements_all.extent(0); ++irec) {
          const int ispec = h_elements_all(irec);
          if (ispec < 0)
            continue;
          if (TagsType{}.has(tag_views(ispec)...))
            h_view(idx++) = get_value(irec, ispec);
        }
        return h_view;
      };
    };

    h_receiver_elements_ = { make_initializer(
        "receiver_elements_", [](int, int ispec) { return ispec; },
        element_types.medium_tags, element_types.property_tags,
        element_types.attenuation_tags) };
    h_receiver_indices_ = { make_initializer(
        "receiver_indices_", [](int irec, int) { return irec; },
        element_types.medium_tags, element_types.property_tags,
        element_types.attenuation_tags) };

    // ── mirror to device ──────────────────────────────────────────────────
    receiver_elements_ = specfem::tag_dispatch::create_mirror_storage_and_copy(
        Kokkos::DefaultExecutionSpace{}, h_receiver_elements_);
    receiver_indices_ = specfem::tag_dispatch::create_mirror_storage_and_copy(
        Kokkos::DefaultExecutionSpace{}, h_receiver_indices_);
  }

public:
  /**
   * @brief Device-side (element, receiver) index views for a material.
   *
   * @param medium      Medium tag.
   * @param property    Property tag.
   * @param attenuation Attenuation tag.
   * @return Tuple of (receiver_elements, receiver_indices), both as
   *         device-accessible views.
   */
  std::tuple<IndexViewType, IndexViewType> get_indices_on_device(
      const specfem::element::medium_tag medium,
      const specfem::element::property_tag property,
      const specfem::element::attenuation_tag attenuation) const {
    return std::make_tuple(
        receiver_elements_.get(medium, property, attenuation),
        receiver_indices_.get(medium, property, attenuation));
  }

  /**
   * @brief Host-side (element, receiver) index views for a material.
   *
   * @param medium      Medium tag.
   * @param property    Property tag.
   * @param attenuation Attenuation tag.
   * @return Tuple of (receiver_elements, receiver_indices), both as
   *         host-accessible views.
   */
  std::tuple<HostIndexViewType, HostIndexViewType> get_indices_on_host(
      const specfem::element::medium_tag medium,
      const specfem::element::property_tag property,
      const specfem::element::attenuation_tag attenuation) const {
    return std::make_tuple(
        h_receiver_elements_.get(medium, property, attenuation),
        h_receiver_indices_.get(medium, property, attenuation));
  }
};

} // namespace specfem::assembly::receivers_impl
