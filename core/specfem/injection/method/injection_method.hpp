#pragma once

#include "specfem/injection/method/coupling_method.hpp"
#include "specfem/injection/points/points.hpp"
#include "specfem/injection/session/provider_session.hpp"

namespace specfem {
namespace injection {

/**
 * @brief An injection method, as built and set up by the program.
 *
 * Adds to the solver-facing specfem::injection::CouplingMethod what the
 * program needs before time stepping: the quantities the method requests from
 * the plugin, and setup(), which hands over the plugin's provider session.
 *
 * @tparam DimensionTag Spatial dimension
 */
template <specfem::element::dimension_tag DimensionTag>
class InjectionMethod : public CouplingMethod<DimensionTag> {
public:
  /**
   * @brief Quantities the method requests per material system.
   * @return Policy for specfem::injection::make_injection_points
   */
  [[nodiscard]] virtual QuantityPolicy quantities() const = 0;

  /**
   * @brief Take ownership of the plugin's session, set up with this method's
   * points.
   *
   * @param assembly Assembly the method will be applied to
   * @param session Provider session, already set up
   */
  virtual void setup(const specfem::assembly::assembly<DimensionTag> &assembly,
                     ProviderSession session) = 0;
};

} // namespace injection
} // namespace specfem
