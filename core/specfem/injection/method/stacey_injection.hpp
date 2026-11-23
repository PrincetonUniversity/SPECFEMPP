#pragma once

#include "specfem/element.hpp"
#include "specfem/injection/method/point_context.hpp"
#include "specfem/injection/method/quantity_list.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <string_view>

namespace specfem {
namespace injection {

/**
 * @brief Stacey injection: the Stacey boundary absorbs the outgoing field and
 * lets the incident field in.
 *
 * At a Stacey point the stiffness kernel adds \f$ -Z(v)\,dS \f$. This method
 * adds
 * \f[
 *   f = \left(t^* + Z(v^*)\right) dS, \qquad
 *   Z(v) = (\rho v_p - \rho v_s)(v \cdot \hat{n})\,\hat{n} + \rho v_s\, v,
 * \f]
 * so the boundary force is \f$ t^* - Z(v - v^*) \f$: the boundary absorbs
 * only the field scattered away from the incident one. \f$ v^* \f$ and
 * \f$ t^* = \sigma^* \hat{n} \f$ are the incident velocity and traction from
 * the plugin, on the point's normal out of the total-field region.
 *
 * Anisotropic points use the Voigt-averaged impedances, as Stacey's absorbing
 * term does; their traction is computed by the plugin from its own model.
 *
 * The caller is responsible for applying the method only at points on a
 * Stacey boundary: elsewhere there is no absorbing term to pair with.
 */
struct StaceyInjection : PointwiseMethodBase<StaceyInjection> {
  /// Point acceleration of an elastic material system
  template <specfem::element::property_tag PropertyTag>
  using Acceleration = specfem::point::acceleration<specfem::tags::Tags<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic, PropertyTag, false>>;

  /// Name selected in the parameter file
  static constexpr std::string_view method_name = "stacey";

  /// Incident quantities requested at elastic points, in slot order
  using elastic_quantities =
      QuantityList<specfem_quantity_velocity, specfem_quantity_traction>;

  /**
   * @brief Injected force at an elastic isotropic or anisotropic point.
   * @tparam PropertyTag Property of the point
   * @param context Point context
   * @return \f$ (t^* + Z(v^*))\,dS \f$
   */
  template <specfem::element::property_tag PropertyTag>
    requires(PropertyTag == specfem::element::property_tag::isotropic ||
             PropertyTag == specfem::element::property_tag::anisotropic)
  KOKKOS_INLINE_FUNCTION static Acceleration<PropertyTag>
  compute(const Context<specfem::element::medium_tag::elastic, PropertyTag>
              &context) {
    const auto v = context.template incident<specfem_quantity_velocity>();
    const auto t = context.template incident<specfem_quantity_traction>();
    const auto &n = context.unit_normal();
    const type_real rho_vp = context.property().rho_vp();
    const type_real rho_vs = context.property().rho_vs();
    const type_real dS = context.weight();

    const type_real vn = v(0) * n(0) + v(1) * n(1) + v(2) * n(2);

    Acceleration<PropertyTag> acceleration;
    for (int i = 0; i < 3; ++i)
      acceleration(i) =
          (t(i) + (rho_vp - rho_vs) * vn * n(i) + rho_vs * v(i)) * dS;
    return acceleration;
  }
};

} // namespace injection
} // namespace specfem
