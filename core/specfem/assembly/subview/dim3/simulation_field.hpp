#pragma once

#include "specfem/assembly/fields.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/enums.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace specfem::assembly {

/**
 * @brief One medium's 3D simulation field, restricted to a range of elements.
 *
 * It is the medium's field (displacement, velocity, acceleration and mass
 * inverse, sharing storage with the parent) together with the slice of the
 * element-to-DOF table for the range. Element `ispec` here is element
 * `range.begin_index() + ispec` of the parent.
 *
 * `index_mapping` is a slice of the medium's `local_index_mapping`, so its
 * values are rows of the medium's field arrays, in [0, nglob). It is a plain
 * view: it can be handed to code that reads tables through their pointer.
 *
 * @tparam SimulationWavefieldType Wavefield of the parent field.
 * @tparam MediumTag Medium the range covers.
 */
template <specfem::simulation::field_type SimulationWavefieldType,
          specfem::element::medium_tag MediumTag>
struct SimulationFieldSubview<specfem::element::dimension_tag::dim3,
                              SimulationWavefieldType, MediumTag>
    : public specfem::assembly::fields_impl::field_impl<
          specfem::element::dimension_tag::dim3, MediumTag> {
private:
  using ParentType =
      specfem::assembly::simulation_field<specfem::element::dimension_tag::dim3,
                                          SimulationWavefieldType>;
  using FieldImplType = specfem::assembly::fields_impl::field_impl<
      specfem::element::dimension_tag::dim3, MediumTag>;

public:
  constexpr static auto simulation_wavefield =
      SimulationWavefieldType; ///< Wavefield type

  /// The medium's device element-to-DOF table in the parent field.
  using LocalTableType = std::remove_cvref_t<
      decltype(std::declval<const ParentType &>()
                   .local_index_mapping.template get<specfem::tags::Tags<
                       specfem::element::dimension_tag::dim3, MediumTag>>())>;

  /// LocalTableType restricted to the range.
  using IndexViewType = decltype(Kokkos::subview(
      std::declval<const LocalTableType &>(), Kokkos::pair<int, int>{},
      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));

  IndexViewType index_mapping; ///< (ispec, iz, iy, ix) -> row of the medium's
                               ///< arrays, ispec counted from the start of the
                               ///< range

  SimulationFieldSubview() = default;

  /**
   * @brief Combine a medium's field with an element slice of the DOF table.
   *
   * @param field The medium's field; shares its storage
   * @param index_mapping Medium-local element-to-DOF table, restricted to the
   * range
   */
  SimulationFieldSubview(const FieldImplType &field,
                         const IndexViewType &index_mapping)
      : FieldImplType(field), index_mapping(index_mapping) {}

  /**
   * @brief Row of a quadrature point in the medium's field arrays.
   *
   * Device only: `index_mapping` is a device view.
   *
   * @param ispec Element, counted from the start of the range
   * @param iz Quadrature point index along z
   * @param iy Quadrature point index along y
   * @param ix Quadrature point index along x
   * @return Index into the medium's field arrays, in [0, nglob)
   */
  KOKKOS_INLINE_FUNCTION int get_iglob(const int &ispec, const int &iz,
                                       const int &iy, const int &ix) const {
    return index_mapping(ispec, iz, iy, ix);
  }
};

/**
 * @brief Restrict a 3D simulation field to a range of elements of one medium.
 *
 * The medium comes from the range's tags, so it cannot disagree with the
 * elements. This is the one place that knows the element axis of
 * `index_mapping`.
 *
 * @tparam SimulationWavefieldType Wavefield of the field.
 * @tparam Tags Tags the range was selected by; must include a medium tag.
 * @param field Field to restrict
 * @param elements Range from `element_types::get_elements_on_device<Tags>()`
 * @return The medium's field with the element slice of its DOF table
 * @throws std::out_of_range If the range is not within the medium's elements
 *
 * @code
 * const auto elements = assembly.element_types.get_elements_on_device<
 *     specfem::tags::Tags<dim3, elastic, isotropic>>();
 * const auto elastic_field = specfem::assembly::subview(field, elements);
 * // device: elastic_field.get_iglob(e, iz, iy, ix) for e in [0, size)
 * @endcode
 */
template <specfem::simulation::field_type SimulationWavefieldType,
          typename Tags>
SimulationFieldSubview<specfem::element::dimension_tag::dim3,
                       SimulationWavefieldType, Tags::medium_tag>
subview(
    const specfem::assembly::simulation_field<
        specfem::element::dimension_tag::dim3, SimulationWavefieldType> &field,
    const specfem::datatype::TaggedElementIndexRange<Tags> &elements) {
  using MediumTags = specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                                         Tags::medium_tag>;

  // The medium's table holds only its own elements, starting at row 0.
  const auto &medium_elements = field.element_ranges.template get<MediumTags>();
  const int begin = elements.begin_index();
  const int end = elements.end_index();
  if (begin < medium_elements.begin_index() || end < begin ||
      end > medium_elements.end_index()) {
    throw std::out_of_range(
        "specfem::assembly::subview: element range [" + std::to_string(begin) +
        ", " + std::to_string(end) + ") is not within the medium's elements [" +
        std::to_string(medium_elements.begin_index()) + ", " +
        std::to_string(medium_elements.end_index()) + ")");
  }

  const int offset = medium_elements.begin_index();
  return { field.field.template get<MediumTags>(),
           Kokkos::subview(field.local_index_mapping.template get<MediumTags>(),
                           Kokkos::make_pair(begin - offset, end - offset),
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL) };
}

} // namespace specfem::assembly
