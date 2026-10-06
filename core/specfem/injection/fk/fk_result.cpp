#include "specfem/injection/fk/fk_result.hpp"

specfem::injection::fk::FkResult::FkResult(int number_of_points,
                                           int coefficient_count,
                                           bool has_traction, bool has_pressure)
    : number_of_points_(number_of_points),
      coefficient_count_(coefficient_count), has_traction_(has_traction),
      has_pressure_(has_pressure) {
  displacement_ = Kokkos::View<type_real ***>(
      "fk_displacement", number_of_points, 3, coefficient_count);
  time_delays_ = Kokkos::View<type_real *>("fk_time_delays", number_of_points);

  if (has_traction) {
    traction_ = Kokkos::View<type_real ***>("fk_traction", number_of_points, 3,
                                            coefficient_count);
  }
  if (has_pressure) {
    pressure_ = Kokkos::View<type_real **>("fk_pressure", number_of_points,
                                           coefficient_count);
  }
}

int specfem::injection::fk::FkResult::number_of_points() const {
  return number_of_points_;
}

int specfem::injection::fk::FkResult::coefficient_count() const {
  return coefficient_count_;
}

int specfem::injection::fk::FkResult::resampling_rate() const {
  return resampling_rate_;
}

type_real specfem::injection::fk::FkResult::resampled_dt() const {
  return resampled_dt_;
}

type_real specfem::injection::fk::FkResult::reference_time() const {
  return reference_time_;
}

bool specfem::injection::fk::FkResult::has_traction() const {
  return has_traction_;
}

bool specfem::injection::fk::FkResult::has_pressure() const {
  return has_pressure_;
}

const Kokkos::View<type_real ***> &
specfem::injection::fk::FkResult::displacement() const {
  return displacement_;
}

Kokkos::View<type_real ***> &specfem::injection::fk::FkResult::displacement() {
  return displacement_;
}

const Kokkos::View<type_real ***> &
specfem::injection::fk::FkResult::traction() const {
  return traction_;
}

Kokkos::View<type_real ***> &specfem::injection::fk::FkResult::traction() {
  return traction_;
}

const Kokkos::View<type_real **> &
specfem::injection::fk::FkResult::pressure() const {
  return pressure_;
}

Kokkos::View<type_real **> &specfem::injection::fk::FkResult::pressure() {
  return pressure_;
}

const Kokkos::View<type_real *> &
specfem::injection::fk::FkResult::time_delays() const {
  return time_delays_;
}

Kokkos::View<type_real *> &specfem::injection::fk::FkResult::time_delays() {
  return time_delays_;
}

void specfem::injection::fk::FkResult::set_sampling(int resampling_rate,
                                                    type_real resampled_dt,
                                                    type_real reference_time) {
  resampling_rate_ = resampling_rate;
  resampled_dt_ = resampled_dt;
  reference_time_ = reference_time;
}
