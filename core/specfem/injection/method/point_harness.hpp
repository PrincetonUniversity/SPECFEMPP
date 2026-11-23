#pragma once

#include "specfem/element.hpp"
#include "specfem/injection/method/point_context.hpp"
#include "specfem/injection/method/quantity_list.hpp"
#include "specfem/injection/session/frame_buffer.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem {
namespace injection {

/**
 * @brief Evaluate an injection method's @c compute at a single point, without
 * a mesh.
 *
 * Lets method authors unit-test their physics: the point is described by
 * explicit values, and @c compute runs on the default execution space as it
 * does in the injection kernel.
 *
 * @code
 * const auto acceleration = specfem::injection::evaluate_point<
 *     MyMethod, specfem::element::medium_tag::elastic,
 *     specfem::element::property_tag::isotropic>(
 *     { kappa, mu, rho }, { 0, 0, 1 }, weight, { vx, vy, vz });
 * @endcode
 *
 * @tparam Method Injection method
 * @tparam MediumTag Medium of the point
 * @tparam PropertyTag Property of the point
 * @param property Material properties at the point
 * @param unit_normal Unit normal, out of the total-field region
 * @param weight Surface quadrature weight dS
 * @param incident Incident values of the method's quantities for the medium,
 * in slot order
 * @param istep Time step seen by @c compute
 * @param time Simulation time seen by @c compute
 * @param parameters Parameters of the method
 * @return The acceleration @c compute returns
 * @throws std::invalid_argument if @p incident does not hold one value per
 * component of the method's quantities
 */
template <typename Method, specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::point::acceleration<specfem::tags::Tags<
    specfem::element::dimension_tag::dim3, MediumTag, PropertyTag, false>>
evaluate_point(
    const typename PointContext<Method, MediumTag, PropertyTag>::properties_type
        &property,
    const std::array<type_real, 3> &unit_normal, const type_real weight,
    const std::vector<type_real> &incident, const int istep = 0,
    const type_real time = 0,
    const typename specfem::injection_impl::MethodParameters<Method>::type
        &parameters = {}) {
  using Context = PointContext<Method, MediumTag, PropertyTag>;
  using PointAccelerationType = specfem::point::acceleration<
      specfem::tags::Tags<specfem::element::dimension_tag::dim3, MediumTag,
                          PropertyTag, false>>;
  constexpr int ncomponents =
      QuantityListTraits<MediumTag, typename Context::quantities>::components;

  if (incident.size() != static_cast<std::size_t>(ncomponents))
    throw std::invalid_argument(
        "specfem::injection::evaluate_point: expected " +
        std::to_string(ncomponents) + " incident values, got " +
        std::to_string(incident.size()) + ".");

  FrameBuffer frame;
  frame.data = FrameBuffer::view_type("specfem::injection::evaluate_point", 1,
                                      ncomponents, 1);
  frame.window_start = istep;
  frame.nsteps = 1;
  const auto h_data = Kokkos::create_mirror_view(frame.data);
  for (int slot = 0; slot < ncomponents; ++slot)
    h_data(0, slot, 0) = incident[slot];
  Kokkos::deep_copy(frame.data, h_data);

  const type_real nx = unit_normal[0];
  const type_real ny = unit_normal[1];
  const type_real nz = unit_normal[2];
  Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace> result(
      "specfem::injection::evaluate_point::result",
      PointAccelerationType::components);

  Kokkos::parallel_for(
      "specfem::injection::evaluate_point",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        const typename Context::template vector_type<3> normal(nx, ny, nz);
        const Context context(frame, 0, istep, time, property, normal, weight,
                              parameters);
        const PointAccelerationType acceleration = Method::compute(context);
        for (int icomp = 0; icomp < PointAccelerationType::components; ++icomp)
          result(icomp) = acceleration(icomp);
      });

  const auto h_result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  PointAccelerationType acceleration;
  for (int icomp = 0; icomp < PointAccelerationType::components; ++icomp)
    acceleration(icomp) = h_result(icomp);
  return acceleration;
}

} // namespace injection
} // namespace specfem
