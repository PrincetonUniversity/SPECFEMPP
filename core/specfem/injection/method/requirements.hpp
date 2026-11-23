#pragma once

#include "specfem/assembly/injection_surface.hpp"
#include "specfem/element.hpp"
#include "specfem/injection/method/point_context.hpp"
#include "specfem/injection/method/quantity_list.hpp"
#include "specfem/point.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <concepts>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

namespace specfem {
namespace injection_impl {

/**
 * @brief Whether an injection method has a @c compute for a material system:
 * a static @c compute taking that system's specfem::injection::PointContext.
 */
template <typename Method, specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
concept ComputesAt = requires(
    const specfem::injection::PointContext<Method, MediumTag, PropertyTag>
        &context) { Method::compute(context); };

/**
 * @brief Whether a method's @c compute for a material system returns the
 * system's point acceleration.
 */
template <typename Method, specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
concept ReturnsAccelerationAt = requires(
    const specfem::injection::PointContext<Method, MediumTag, PropertyTag>
        &context) {
  {
    Method::compute(context)
  } -> std::same_as<specfem::point::acceleration<specfem::tags::Tags<
      specfem::element::dimension_tag::dim3, MediumTag, PropertyTag, false>>>;
};

/// Material systems the injection surface can hold
using InjectionCombinations = std::remove_cvref_t<
    decltype(specfem::assembly::InjectionSurface<
             specfem::element::dimension_tag::dim3>::combinations)>;

/// Tags of the @c I th material system of the injection surface
template <std::size_t I>
using InjectionCombinationTags = specfem::tag_dispatch::impl::combo_to_tags_t<
    InjectionCombinations, I, InjectionCombinations::combo_type::arity>;

/// Whether a method has a @c compute for any material system of a medium
template <typename Method, specfem::element::medium_tag MediumTag>
inline constexpr bool computes_in_medium_v =
    []<std::size_t... Is>(std::index_sequence<Is...>) {
      return ((InjectionCombinationTags<Is>::medium_tag == MediumTag &&
               ComputesAt<Method, InjectionCombinationTags<Is>::medium_tag,
                          InjectionCombinationTags<Is>::property_tag>) ||
              ...);
    }(std::make_index_sequence<InjectionCombinations::size>{});

/// Whether every medium a method computes in declares a quantity list
template <typename Method>
inline constexpr bool computes_only_where_declared_v =
    []<std::size_t... Is>(std::index_sequence<Is...>) {
      return ((!ComputesAt<Method, InjectionCombinationTags<Is>::medium_tag,
                           InjectionCombinationTags<Is>::property_tag> ||
               DeclaresQuantities<Method,
                                  InjectionCombinationTags<Is>::medium_tag>) &&
              ...);
    }(std::make_index_sequence<InjectionCombinations::size>{});

/// Whether every @c compute of a method returns its system's acceleration
template <typename Method>
inline constexpr bool returns_acceleration_v = []<std::size_t... Is>(
                                                   std::index_sequence<Is...>) {
  return (
      (!ComputesAt<Method, InjectionCombinationTags<Is>::medium_tag,
                   InjectionCombinationTags<Is>::property_tag> ||
       ReturnsAccelerationAt<Method, InjectionCombinationTags<Is>::medium_tag,
                             InjectionCombinationTags<Is>::property_tag>) &&
      ...);
}(std::make_index_sequence<InjectionCombinations::size>{});

/// Whether a method has a @c compute for any material system
template <typename Method>
inline constexpr bool computes_somewhere_v =
    []<std::size_t... Is>(std::index_sequence<Is...>) {
      return (ComputesAt<Method, InjectionCombinationTags<Is>::medium_tag,
                         InjectionCombinationTags<Is>::property_tag> ||
              ...);
    }(std::make_index_sequence<InjectionCombinations::size>{});

/**
 * @brief The method declares @c static @c constexpr @c std::string_view
 * @c method_name.
 */
template <typename Method>
concept NamedMethod = requires {
  { Method::method_name } -> std::convertible_to<std::string_view>;
};

/**
 * @brief The method's @c Parameters, if any, can be default-constructed and
 * copied to the device.
 */
template <typename Method>
concept DeviceParameters =
    std::is_trivially_copyable_v<typename MethodParameters<Method>::type> &&
    std::default_initializable<typename MethodParameters<Method>::type>;

/**
 * @brief Some static @c compute(const Context<Medium, Property> &) of the
 * method matches a material system.
 */
template <typename Method>
concept ComputesSomewhere = computes_somewhere_v<Method>;

/// Every @c compute of the method returns its system's point acceleration.
template <typename Method>
concept ComputeReturnsAcceleration = returns_acceleration_v<Method>;

/// Every medium the method computes in has a quantity list.
template <typename Method>
concept ComputeHasQuantities = computes_only_where_declared_v<Method>;

/// A quantity list the method declares has a @c compute in its medium.
template <typename Method, specfem::element::medium_tag MediumTag>
concept QuantitiesHaveCompute = !DeclaresQuantities<Method, MediumTag> ||
                                computes_in_medium_v<Method, MediumTag>;

/// The method's quantity list for a medium lists no quantity twice.
template <typename Method, specfem::element::medium_tag MediumTag>
concept UniqueQuantities = specfem::injection::QuantityListTraits<
    MediumTag, typename MethodQuantities<Method, MediumTag>::type>::unique;

/// The method's quantity list for a medium lists only quantities the medium
/// provides.
template <typename Method, specfem::element::medium_tag MediumTag>
concept SupportedQuantities = specfem::injection::QuantityListTraits<
    MediumTag, typename MethodQuantities<Method, MediumTag>::type>::supported;

/// Requirements on a medium's quantity list.
template <typename Method, specfem::element::medium_tag MediumTag>
concept ValidQuantities = QuantitiesHaveCompute<Method, MediumTag> &&
                          UniqueQuantities<Method, MediumTag> &&
                          SupportedQuantities<Method, MediumTag>;

/**
 * @brief Requirements of specfem::injection::PointwiseSurfaceMethod on the
 * method it wraps.
 *
 * Checked in order, and the compiler reports the first one not satisfied.
 */
template <typename Method>
concept PointwiseInjectionMethod =
    NamedMethod<Method> && DeviceParameters<Method> &&
    ComputesSomewhere<Method> && ComputeReturnsAcceleration<Method> &&
    ComputeHasQuantities<Method> &&
    ValidQuantities<Method, specfem::element::medium_tag::elastic> &&
    ValidQuantities<Method, specfem::element::medium_tag::acoustic>;

} // namespace injection_impl
} // namespace specfem
