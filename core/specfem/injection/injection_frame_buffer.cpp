#include "specfem/injection/injection_frame_buffer.hpp"

specfem::injection::InjectionFrameBuffer::InjectionFrameBuffer(
    int number_of_points, int number_of_components, int number_of_steps)
    : values_("injection_frame_buffer", number_of_points, number_of_components,
              number_of_steps),
      number_of_points_(number_of_points),
      number_of_components_(number_of_components),
      number_of_steps_(number_of_steps) {}

int specfem::injection::InjectionFrameBuffer::number_of_points() const {
  return number_of_points_;
}

int specfem::injection::InjectionFrameBuffer::number_of_components() const {
  return number_of_components_;
}

int specfem::injection::InjectionFrameBuffer::number_of_steps() const {
  return number_of_steps_;
}

const Kokkos::View<type_real ***> &
specfem::injection::InjectionFrameBuffer::values() const {
  return values_;
}

Kokkos::View<type_real ***> &
specfem::injection::InjectionFrameBuffer::values() {
  return values_;
}
