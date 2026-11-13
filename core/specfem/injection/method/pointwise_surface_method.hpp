#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/injection/method/compute_injection_interaction.hpp"
#include "specfem/injection/method/injection_method.hpp"
#include "specfem/injection/method/point_context.hpp"
#include "specfem/injection/method/quantity_list.hpp"
#include "specfem/injection/method/requirements.hpp"
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
namespace injection {

/**
 * @brief Injection method that adds a term computed pointwise on the
 * injection surface, built from a method that provides only:
 * - @c method_name, the name selected in the parameter file;
 * - @c elastic_quantities and/or @c acoustic_quantities, the
 *   specfem::injection::QuantityList it requests per medium;
 * - a static @c compute(const PointContext<Method, MediumTag, PropertyTag> &)
 *   per supported material system, returning the point acceleration to add;
 * - optionally @c Parameters, a trivially copyable struct available in
 *   @c compute through @c context.parameters(). It is default-constructed.
 *
 * It requests the quantities from the plugin, keeps the provider session, and
 * runs the injection kernel. The material systems a method supports are those
 * it has a @c compute for. The method must satisfy
 * specfem::injection_impl::PointwiseInjectionMethod; the compiler names the
 * first requirement it does not meet.
 *
 * Deriving the method from specfem::injection::PointwiseMethodBase provides
 * the shorthand @c Context<MediumTag, PropertyTag>:
 *
 * @code
 * struct MyMethod : specfem::injection::PointwiseMethodBase<MyMethod> {
 *   static constexpr std::string_view method_name = "my-method";
 *   using elastic_quantities = QuantityList<specfem_quantity_velocity>;
 *
 *   KOKKOS_INLINE_FUNCTION static auto
 *   compute(const Context<specfem::element::medium_tag::elastic,
 *                         specfem::element::property_tag::isotropic> &ctx);
 * };
 *
 * registry.add<specfem::injection::PointwiseSurfaceMethod<MyMethod>>();
 * @endcode
 *
 * @tparam Method The pointwise method
 */
template <specfem::injection_impl::PointwiseInjectionMethod Method>
class PointwiseSurfaceMethod
    : public InjectionMethod<specfem::element::dimension_tag::dim3> {
public:
  /// Material systems the injection surface can hold
  constexpr static auto combinations = specfem::assembly::InjectionSurface<
      specfem::element::dimension_tag::dim3>::combinations;

  [[nodiscard]] std::string_view name() const override {
    return Method::method_name;
  }

  [[nodiscard]] QuantityPolicy quantities() const override {
    return [](const specfem::element::medium_tag medium,
              const specfem::element::property_tag property)
               -> std::optional<std::vector<QuantityRequest>> {
      namespace impl = specfem::injection_impl;
      std::optional<std::vector<QuantityRequest>> requests;
      specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
        constexpr auto M = TagsType::medium_tag;
        constexpr auto P = TagsType::property_tag;
        if constexpr (impl::ComputesAt<Method, M, P>) {
          if (M == medium && P == property)
            requests = to_requests<M>(
                typename impl::MethodQuantities<Method, M>::type{});
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
      throw std::runtime_error("Injection method '" + std::string(name()) +
                               "' applies to the forward wavefield only.");
    namespace impl = specfem::injection_impl;
    const auto &frame = session().frame();
    const typename impl::MethodParameters<Method>::type parameters{};
    specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
      constexpr auto M = TagsType::medium_tag;
      constexpr auto P = TagsType::property_tag;
      if constexpr (impl::ComputesAt<Method, M, P>) {
        if (M == medium)
          impl::compute_injection_interaction<Method, M, P>(assembly, frame,
                                                            parameters, istep);
      }
    });
  }

private:
  ProviderSession &session() {
    if (!session_)
      throw std::logic_error("Injection method '" + std::string(name()) +
                             "' used before setup.");
    return *session_;
  }

  std::optional<ProviderSession> session_; ///< Plugin session, after setup
};

} // namespace injection
} // namespace specfem
