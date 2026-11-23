#pragma once

#include "specfem/element.hpp"
#include "specfem/injection/session/points.hpp"
#include <array>
#include <cstddef>
#include <vector>

namespace specfem {
namespace injection {

/**
 * @brief Quantities an injection method requests from the plugin for one
 * medium, in slot order.
 *
 * @code
 * using elastic_quantities =
 *     QuantityList<specfem_quantity_velocity, specfem_quantity_traction>;
 * @endcode
 *
 * @tparam Quantities Requested quantities
 */
template <specfem_quantity... Quantities> struct QuantityList {};

/**
 * @brief Buffer components a quantity occupies at a point of a medium.
 *
 * Elastic points take displacement, velocity, acceleration and traction as 3D
 * vectors; acoustic points take the potential rate as a scalar.
 *
 * @tparam MediumTag Medium of the point
 * @param quantity Requested quantity
 * @return Number of components, or 0 if the medium does not support it
 */
template <specfem::element::medium_tag MediumTag>
constexpr int quantity_components(const specfem_quantity quantity) {
  if constexpr (MediumTag == specfem::element::medium_tag::elastic) {
    switch (quantity) {
    case specfem_quantity_displacement:
    case specfem_quantity_velocity:
    case specfem_quantity_acceleration:
    case specfem_quantity_traction:
      return specfem::element::attributes<
          specfem::element::dimension_tag::dim3,
          specfem::element::medium_tag::elastic>::components;
    default:
      return 0;
    }
  } else if constexpr (MediumTag == specfem::element::medium_tag::acoustic) {
    return quantity == specfem_quantity_potential_rate ? 1 : 0;
  } else {
    return 0;
  }
}

/**
 * @brief Position of a quantity in a medium's quantity list.
 *
 * @tparam MediumTag Medium of the list
 * @tparam Quantity Quantity looked up
 * @tparam List specfem::injection::QuantityList
 */
template <specfem::element::medium_tag MediumTag, specfem_quantity Quantity,
          typename List>
struct QuantitySlot;

template <specfem::element::medium_tag MediumTag, specfem_quantity Quantity,
          specfem_quantity... Quantities>
struct QuantitySlot<MediumTag, Quantity, QuantityList<Quantities...>> {
  /// Whether the list contains the quantity
  constexpr static bool listed = ((Quantities == Quantity) || ...);

  /// Components the quantity occupies
  constexpr static int components = quantity_components<MediumTag>(Quantity);

  /// First buffer component of the quantity: the components of the
  /// quantities listed before it
  constexpr static int offset = [] {
    int offset = 0;
    bool found = false;
    ((found = found || Quantities == Quantity,
      offset += found ? 0 : quantity_components<MediumTag>(Quantities)),
     ...);
    return offset;
  }();
};

/**
 * @brief Properties of a medium's quantity list.
 *
 * @tparam MediumTag Medium of the list
 * @tparam List specfem::injection::QuantityList
 */
template <specfem::element::medium_tag MediumTag, typename List>
struct QuantityListTraits;

template <specfem::element::medium_tag MediumTag,
          specfem_quantity... Quantities>
struct QuantityListTraits<MediumTag, QuantityList<Quantities...>> {
  /// Whether the medium supports every quantity of the list
  constexpr static bool supported =
      ((quantity_components<MediumTag>(Quantities) > 0) && ...);

  /// Whether no quantity is listed twice
  constexpr static bool unique = [] {
    const std::array<specfem_quantity, sizeof...(Quantities)> quantities{
      Quantities...
    };
    for (std::size_t i = 0; i < quantities.size(); ++i)
      for (std::size_t j = i + 1; j < quantities.size(); ++j)
        if (quantities[i] == quantities[j])
          return false;
    return true;
  }();

  /// Buffer components of the whole list
  constexpr static int components =
      (0 + ... + quantity_components<MediumTag>(Quantities));
};

/**
 * @brief The plugin request for a medium's quantity list.
 *
 * @tparam MediumTag Medium of the list
 * @tparam Quantities Quantities of the list
 * @return One request per quantity, in slot order
 */
template <specfem::element::medium_tag MediumTag,
          specfem_quantity... Quantities>
std::vector<QuantityRequest> to_requests(QuantityList<Quantities...>) {
  return { QuantityRequest{ Quantities,
                            quantity_components<MediumTag>(Quantities) }... };
}

} // namespace injection

namespace injection_impl {

/**
 * @brief Whether an injection method declares a quantity list for a medium:
 * @c Method::elastic_quantities or @c Method::acoustic_quantities.
 *
 * @tparam Method Injection method
 * @tparam MediumTag Medium
 */
template <typename Method, specfem::element::medium_tag MediumTag>
concept DeclaresQuantities =
    (MediumTag == specfem::element::medium_tag::elastic && requires {
      typename Method::elastic_quantities;
    }) || (MediumTag == specfem::element::medium_tag::acoustic && requires {
      typename Method::acoustic_quantities;
    });

/**
 * @brief The quantity list an injection method declares for a medium, empty
 * if absent.
 *
 * @tparam Method Injection method
 * @tparam MediumTag Medium
 */
template <typename Method, specfem::element::medium_tag MediumTag>
struct MethodQuantities {
  using type = specfem::injection::QuantityList<>;
};

template <typename Method>
  requires DeclaresQuantities<Method, specfem::element::medium_tag::elastic>
struct MethodQuantities<Method, specfem::element::medium_tag::elastic> {
  using type = typename Method::elastic_quantities;
};

template <typename Method>
  requires DeclaresQuantities<Method, specfem::element::medium_tag::acoustic>
struct MethodQuantities<Method, specfem::element::medium_tag::acoustic> {
  using type = typename Method::acoustic_quantities;
};

} // namespace injection_impl
} // namespace specfem
