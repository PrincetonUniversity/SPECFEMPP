#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/injection/method/compute_injection_interaction.hpp"
#include "specfem/injection/method/injection_method.hpp"
#include "specfem/injection/method/point_context.hpp"
#include "specfem/injection/method/quantity_list.hpp"
#include "specfem/point.hpp"
#include "specfem/simulation.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <concepts>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace specfem {
namespace injection_impl {

/**
 * @brief Whether an injection method computes its term at points of a
 * material system: it has a @c compute overload taking that system's
 * specfem::injection::PointContext.
 */
template <typename Method, specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
concept computes_at = requires(
    const specfem::injection::PointContext<Method, MediumTag, PropertyTag>
        &context) { Method::compute(context); };

/**
 * @brief Whether a method's @c compute for a material system returns the
 * system's point acceleration.
 */
template <typename Method, specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
constexpr bool returns_acceleration() {
  using Context =
      specfem::injection::PointContext<Method, MediumTag, PropertyTag>;
  return std::same_as<
      decltype(Method::compute(std::declval<const Context &>())),
      specfem::point::acceleration<
          specfem::tags::Tags<specfem::element::dimension_tag::dim3, MediumTag,
                              PropertyTag, false>>>;
}

} // namespace injection_impl

namespace injection {

/**
 * @brief Base of injection methods that add a term computed pointwise on the
 * injection surface.
 *
 * A method derives from it and provides only:
 * - @c method_name, the name selected in the parameter file;
 * - @c elastic_quantities and/or @c acoustic_quantities, the
 *   specfem::injection::QuantityList it requests per medium;
 * - a static @c compute(const Context<MediumTag, PropertyTag> &) per supported
 *   material system, returning the point acceleration to add;
 * - optionally @c Parameters, a trivially copyable struct available in
 *   @c compute through @c context.parameters(). It is default-constructed.
 *
 * The base requests the quantities from the plugin, keeps the provider
 * session, and runs the injection kernel. The material systems a method
 * supports are those it has a @c compute for.
 *
 * @code
 * struct MyMethod : specfem::injection::PointwiseSurfaceMethod<MyMethod> {
 *   static constexpr std::string_view method_name = "my-method";
 *   using elastic_quantities = QuantityList<specfem_quantity_velocity>;
 *
 *   KOKKOS_INLINE_FUNCTION static auto
 *   compute(const Context<specfem::element::medium_tag::elastic,
 *                         specfem::element::property_tag::isotropic> &ctx);
 * };
 * @endcode
 *
 * @tparam Derived The injection method
 */
template <typename Derived>
class PointwiseSurfaceMethod
    : public InjectionMethod<specfem::element::dimension_tag::dim3> {
public:
  /// Point context of a material system, the argument of @c compute
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  using Context = PointContext<Derived, MediumTag, PropertyTag>;

  /// Material systems the injection surface can hold
  constexpr static auto combinations = specfem::assembly::InjectionSurface<
      specfem::element::dimension_tag::dim3>::combinations;

  PointwiseSurfaceMethod() {
    using parameters_type =
        typename specfem::injection_impl::MethodParameters<Derived>::type;
    static_assert(std::is_trivially_copyable_v<parameters_type>,
                  "Injection method Parameters must be trivially copyable: "
                  "they are copied to the device");
    specfem::tag_dispatch::for_each(combinations, []<typename TagsType>() {
      constexpr auto medium = TagsType::medium_tag;
      constexpr auto property = TagsType::property_tag;
      if constexpr (specfem::injection_impl::computes_at<Derived, medium,
                                                         property>) {
        static_assert(
            specfem::injection_impl::returns_acceleration<Derived, medium,
                                                          property>(),
            "Injection method compute must return the point "
            "acceleration of its material system");
        static_assert(
            QuantityListTraits<
                medium, typename specfem::injection_impl::MethodQuantities<
                            Derived, medium>::type>::supported,
            "Injection method requests a quantity its medium does not "
            "support");
      }
    });
  }

  [[nodiscard]] std::string_view name() const override {
    return Derived::method_name;
  }

  [[nodiscard]] QuantityPolicy quantities() const override {
    return [](const specfem::element::medium_tag medium,
              const specfem::element::property_tag property)
               -> std::optional<std::vector<QuantityRequest>> {
      std::optional<std::vector<QuantityRequest>> requests;
      specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
        constexpr auto M = TagsType::medium_tag;
        constexpr auto P = TagsType::property_tag;
        if constexpr (specfem::injection_impl::computes_at<Derived, M, P>) {
          if (M == medium && P == property)
            requests = to_requests<M>(
                typename specfem::injection_impl::MethodQuantities<Derived,
                                                                   M>::type{});
        }
      });
      return requests;
    };
  }

  void
  setup(const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
            &,
        ProviderSession session) override {
    session_.emplace(std::move(session));
  }

  void prepare_step(const int istep) override {
    session().ensure_window(istep);
  }

  void apply(specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
                 &assembly,
             const specfem::element::medium_tag medium,
             const specfem::simulation::field_type wavefield,
             const int istep) override {
    if (wavefield != specfem::simulation::field_type::forward)
      throw std::runtime_error("Injection method '" +
                               std::string(Derived::method_name) +
                               "' applies to the forward wavefield only.");
    const auto &frame = session().frame();
    const typename specfem::injection_impl::MethodParameters<Derived>::type
        parameters{};
    specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
      constexpr auto M = TagsType::medium_tag;
      constexpr auto P = TagsType::property_tag;
      if constexpr (specfem::injection_impl::computes_at<Derived, M, P>) {
        if (M == medium)
          specfem::injection_impl::compute_injection_interaction<Derived, M, P>(
              assembly, frame, parameters, istep);
      }
    });
  }

private:
  ProviderSession &session() {
    if (!session_)
      throw std::logic_error("Injection method '" +
                             std::string(Derived::method_name) +
                             "' used before setup.");
    return *session_;
  }

  std::optional<ProviderSession> session_; ///< Plugin session, after setup
};

} // namespace injection
} // namespace specfem
