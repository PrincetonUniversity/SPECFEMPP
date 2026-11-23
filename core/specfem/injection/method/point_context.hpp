#pragma once

#include "specfem/datatype/register_array.hpp"
#include "specfem/element.hpp"
#include "specfem/injection/method/quantity_list.hpp"
#include "specfem/injection/session/frame_buffer.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection_impl {

/** @brief Parameters of a method that declares none. */
struct NoParameters {};

/**
 * @brief The parameters of an injection method: @c Method::Parameters, or
 * specfem::injection_impl::NoParameters if absent.
 *
 * @tparam Method Injection method
 */
template <typename Method> struct MethodParameters {
  using type = NoParameters;
};

template <typename Method>
  requires requires { typename Method::Parameters; }
struct MethodParameters<Method> {
  using type = typename Method::Parameters;
};

} // namespace injection_impl

namespace injection {

/**
 * @brief What an injection method's @c compute sees at one injection point.
 *
 * Built by the injection kernel for every entry of the injection surface.
 * Incident values are read from the plugin's frame when incident() is called;
 * material properties are those of the solver at the point.
 *
 * @tparam Method Injection method
 * @tparam MediumTag Medium of the point
 * @tparam PropertyTag Property of the point
 */
template <typename Method, specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
class PointContext {
public:
  /// Quantities the method requests for this medium, in slot order
  using quantities =
      typename specfem::injection_impl::MethodQuantities<Method,
                                                         MediumTag>::type;
  /// Parameters of the method
  using parameters_type =
      typename specfem::injection_impl::MethodParameters<Method>::type;
  /// Material properties at the point
  using properties_type = specfem::point::properties<specfem::tags::Tags<
      specfem::element::dimension_tag::dim3, MediumTag, PropertyTag, false>>;
  /// Small vector of @c N values
  template <int N>
  using vector_type = specfem::datatype::RegisterArray<
      type_real, Kokkos::extents<std::size_t, N>, Kokkos::layout_left>;

  /**
   * @brief Context of one injection point.
   *
   * @param frame Incident field of the current window
   * @param row Row of the point in the plugin's point list
   * @param istep Current time step
   * @param time Simulation time of @p istep
   * @param property Material properties at the point
   * @param normal Unit normal, out of the total-field region
   * @param weight Surface quadrature weight dS
   * @param parameters Parameters of the method
   */
  KOKKOS_INLINE_FUNCTION
  PointContext(const FrameBuffer &frame, const int row, const int istep,
               const type_real time, const properties_type &property,
               const vector_type<3> &normal, const type_real weight,
               const parameters_type &parameters)
      : frame_(frame), row_(row), istep_(istep), time_(time),
        property_(property), normal_(normal), weight_(weight),
        parameters_(parameters) {}

  /**
   * @brief Incident value of a requested quantity at the point.
   *
   * @tparam Quantity A quantity in the method's list for this medium
   * @return Its components
   */
  template <specfem_quantity Quantity>
  KOKKOS_INLINE_FUNCTION auto incident() const {
    using Slot = QuantitySlot<MediumTag, Quantity, quantities>;
    static_assert(Slot::listed,
                  "PointContext::incident: the quantity is not in the "
                  "method's quantity list for this medium");
    static_assert(Slot::components > 0,
                  "PointContext::incident: the medium does not support the "
                  "quantity");
    vector_type<Slot::components> values;
    for (int icomp = 0; icomp < Slot::components; ++icomp)
      values(icomp) = frame_.load(istep_, row_, Slot::offset + icomp);
    return values;
  }

  /**
   * @brief Material properties at the point.
   * @return Point properties, as used by the solver's kernels
   */
  KOKKOS_INLINE_FUNCTION const properties_type &property() const {
    return property_;
  }

  /**
   * @brief Unit normal at the point, out of the total-field region.
   * @return Normal vector
   */
  KOKKOS_INLINE_FUNCTION const vector_type<3> &unit_normal() const {
    return normal_;
  }

  /**
   * @brief Surface quadrature weight dS of the point.
   * @return Weight
   */
  KOKKOS_INLINE_FUNCTION type_real weight() const { return weight_; }

  /**
   * @brief Simulation time of the current step.
   * @return @f$ t_0 + \text{istep} \, \Delta t @f$
   */
  KOKKOS_INLINE_FUNCTION type_real time() const { return time_; }

  /**
   * @brief Current time step.
   * @return Step index
   */
  KOKKOS_INLINE_FUNCTION int istep() const { return istep_; }

  /**
   * @brief Parameters of the method.
   * @return @c Method::Parameters
   */
  KOKKOS_INLINE_FUNCTION const parameters_type &parameters() const
    requires requires { typename Method::Parameters; }
  {
    return parameters_;
  }

private:
  const FrameBuffer &frame_;          ///< Incident field
  int row_;                           ///< Plugin row of the point
  int istep_;                         ///< Current time step
  type_real time_;                    ///< Simulation time
  const properties_type &property_;   ///< Material properties
  vector_type<3> normal_;             ///< Unit normal
  type_real weight_;                  ///< Surface weight
  const parameters_type &parameters_; ///< Method parameters
};

/**
 * @brief Optional base of a pointwise injection method, providing the
 * shorthand @c Context<MediumTag, PropertyTag> for the method's
 * specfem::injection::PointContext.
 *
 * It only names types, so it is safe to derive from while the method is
 * still being defined; the method's requirements are checked when it is
 * wrapped in specfem::injection::PointwiseSurfaceMethod.
 *
 * @tparam Method The injection method deriving from it
 */
template <typename Method> struct PointwiseMethodBase {
  /// Point context of a material system, the argument of @c compute
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  using Context = PointContext<Method, MediumTag, PropertyTag>;
};

} // namespace injection
} // namespace specfem
