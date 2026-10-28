#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/injection/method/point_context.hpp"
#include "specfem/injection/session/frame_buffer.hpp"
#include "specfem/point.hpp"
#include "specfem/simulation.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection_impl {

/**
 * @brief Add an injection method's term at the injection points of one
 * material system to the forward field's acceleration.
 *
 * For every entry of the injection surface in the system, builds the point's
 * specfem::injection::PointContext, calls @c Method::compute and adds the
 * result atomically: points shared by several elements or faces receive one
 * contribution per entry. Points where the acceleration is held at zero
 * (acoustic free surface) receive nothing.
 *
 * @tparam Method Injection method
 * @tparam MediumTag Medium of the system
 * @tparam PropertyTag Property of the system
 * @param assembly Assembly being time-stepped
 * @param frame Incident field window holding @p istep
 * @param parameters Parameters of the method
 * @param istep Current time step
 */
template <typename Method, specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void compute_injection_interaction(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::injection::FrameBuffer &frame,
    const typename MethodParameters<Method>::type &parameters,
    const int istep) {
  using Context =
      specfem::injection::PointContext<Method, MediumTag, PropertyTag>;
  using PointTags = specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                                        MediumTag, PropertyTag, false>;
  using PointAccelerationType = specfem::point::acceleration<PointTags>;
  using PointIndexType =
      specfem::point::index<specfem::element::dimension_tag::dim3, false>;

  const auto &surface = assembly.injection_surface;
  const auto entries = surface.template get_entries<MediumTag, PropertyTag>();
  if (entries.size() == 0)
    return;

  const auto ispec = surface.ispec;
  const auto iz = surface.iz;
  const auto iy = surface.iy;
  const auto ix = surface.ix;
  const auto row = surface.row;
  const auto normal = surface.normal;
  const auto weight = surface.weight;
  const auto boundary_tag = surface.boundary_tag;
  const auto &properties = assembly.properties;
  const auto &field = assembly.fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();
  const type_real time = assembly.t0 + istep * assembly.dt;

  Kokkos::Profiling::pushRegion("Compute Injection Interaction");

  Kokkos::parallel_for(
      "specfem::injection::compute_injection_interaction",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(entries.begin_index(),
                                                         entries.end_index()),
      KOKKOS_LAMBDA(const int entry) {
        const auto tag = boundary_tag(entry);
        if (tag == specfem::element::boundary_tag::acoustic_free_surface ||
            tag == specfem::element::boundary_tag::composite_stacey_dirichlet)
          return;

        const PointIndexType index(ispec(entry), iz(entry), iy(entry),
                                   ix(entry));
        typename Context::properties_type property;
        specfem::assembly::load_on_device(index, properties, property);

        const typename Context::template vector_type<3> unit_normal(
            normal(entry, 0), normal(entry, 1), normal(entry, 2));
        const Context context(frame, row(entry), istep, time, property,
                              unit_normal, weight(entry), parameters);

        const PointAccelerationType acceleration = Method::compute(context);
        specfem::assembly::atomic_add_on_device(index, field, acceleration);
      });

  Kokkos::Profiling::popRegion();
}

} // namespace injection_impl
} // namespace specfem
