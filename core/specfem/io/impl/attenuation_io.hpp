#pragma once

#include "specfem/element.hpp"
#include <string>

namespace specfem {
namespace io {
namespace impl {

/**
 * @brief Attenuation-aware property I/O hook for the property writer/reader.
 *
 * The property buffer stores the runtime (unrelaxed) moduli `kappa*scale`,
 * `mu*scale`. For a portable model file we instead persist the PHYSICAL
 * (relaxed) moduli plus the per-medium attenuation datasets (e.g. Qkappa/Qmu).
 * This hook bridges the two representations for attenuating (medium, property)
 * combinations:
 *  - @ref write_view un-scales a scaled modulus (`physical = unrelaxed /
 * scale`) into a fresh scratch view -- the live property buffer is never
 * mutated, which matters because on a CPU build the host mirror aliases the
 * device view;
 *  - @ref attenuation_append writes the per-medium model-I/O datasets; and
 *  - @ref attenuation_read reads them back.
 *
 * The read-side transformation (re-scaling the physical moduli to unrelaxed and
 * recomputing scale factors / relaxation rates) lives in the attenuation
 * container's `recompute()`, which the reader calls after @ref
 * attenuation_read.
 *
 * This struct is attenuation-type agnostic: it holds no knowledge of which
 * datasets a medium persists or which property views are scaled. All of that
 * lives in the per-(medium, property, attenuation) `attenuation_medium`
 * container, which exposes @c has_attenuating_elements, @c is_scaled_property,
 * @c for_each_io_host_view and @c scale_into. Adding a new attenuation type is
 * therefore a new container specialization with no change here. For
 * non-attenuating media every method is a compile-time no-op.
 *
 * @tparam AttenuationType  The assembly Attenuation<DimensionTag> type.
 * @tparam ElementTypesType The assembly element_types<DimensionTag> type.
 */
template <typename AttenuationType, typename ElementTypesType>
struct AttenuationIO {
  const AttenuationType &attenuation;    ///< Assembly attenuation container
  const ElementTypesType &element_types; ///< Assembly element-type organization

  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag, typename GroupType,
            typename ViewType>
  void write_view(GroupType &group, const std::string &name,
                  const ViewType &view) const {
    if constexpr (AttenuationType::template has_attenuation<MediumTag,
                                                            PropertyTag>()) {
      const auto &c =
          attenuation.template get_container<MediumTag, PropertyTag>();
      if (c.has_attenuating_elements() && c.is_scaled_property(name)) {
        // Un-scale into a fresh scratch view; never mutate the live buffer
        // (NOT create_mirror_view, which aliases the device view on CPU).
        ViewType scratch("AttenuationIO_scratch", view.get_mapping());
        c.scale_into(
            scratch, view, name, /*to_physical=*/true,
            element_types.get_elements_on_host(MediumTag, PropertyTag));
        group.createDataset(name, scratch).write();
        return;
      }
    }
    group.createDataset(name, view).write();
  }

  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag, typename GroupType>
  void attenuation_append(GroupType &group) const {
    if constexpr (AttenuationType::template has_attenuation<MediumTag,
                                                            PropertyTag>()) {
      const auto &c =
          attenuation.template get_container<MediumTag, PropertyTag>();
      if (!c.has_attenuating_elements())
        return;
      c.for_each_io_host_view([&](const auto &view, const std::string &name) {
        group.createDataset(name, view).write();
      });
    }
  }

  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag, typename GroupType>
  void attenuation_read(GroupType &group) const {
    if constexpr (AttenuationType::template has_attenuation<MediumTag,
                                                            PropertyTag>()) {
      const auto &c =
          attenuation.template get_container<MediumTag, PropertyTag>();
      if (!c.has_attenuating_elements())
        return;
      c.for_each_io_host_view([&](const auto &view, const std::string &name) {
        group.openDataset(name, view).read();
      });
    }
  }
};

} // namespace impl
} // namespace io
} // namespace specfem
