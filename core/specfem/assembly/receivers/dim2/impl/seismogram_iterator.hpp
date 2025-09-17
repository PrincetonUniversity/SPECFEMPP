#pragma once

#include "enumerations/interface.hpp"
#include <array>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace specfem::assembly::receivers_impl {

//
// DIM2 SPECIALIZATION
//
template <> class SeismogramIterator<specfem::dimension::type::dim2> {
private:
  static constexpr int ncomponents = 2;

  class Iterator {
  public:
    Iterator(const int irec, const int iseis, const int seis_step,
             const type_real dt, const type_real t0,
             const int nstep_between_samples,
             Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>
                 h_sine_receiver_angle,
             Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>
                 h_cosine_receiver_angle,
             Kokkos::View<type_real ***[2], Kokkos::LayoutLeft,
                          Kokkos::DefaultHostExecutionSpace>
                 seismogram_components)
        : irec(irec), iseis(iseis), seis_step(seis_step), dt(dt), t0(t0),
          nstep_between_samples(nstep_between_samples),
          h_sine_receiver_angle(h_sine_receiver_angle),
          h_cosine_receiver_angle(h_cosine_receiver_angle),
          seismo_components(seismogram_components) {}

    std::tuple<type_real, std::array<type_real, 2> > operator*() {
      type_real time = seis_step * dt * nstep_between_samples + t0;

      std::array<type_real, 2> seismograms;

      // 2D: Use angle-based rotation
      seismograms[0] = h_cosine_receiver_angle(irec) *
                           seismo_components(seis_step, iseis, irec, 0) -
                       h_sine_receiver_angle(irec) *
                           seismo_components(seis_step, iseis, irec, 1);

      seismograms[1] = h_sine_receiver_angle(irec) *
                           seismo_components(seis_step, iseis, irec, 0) +
                       h_cosine_receiver_angle(irec) *
                           seismo_components(seis_step, iseis, irec, 1);

      return std::make_tuple(time, seismograms);
    }

    Iterator &operator++() {
      ++seis_step;
      return *this;
    }

    bool operator==(const Iterator &other) const {
      return seis_step == other.seis_step;
    }

    bool operator!=(const Iterator &other) const {
      return seis_step != other.seis_step;
    }

  private:
    int irec;
    int iseis;
    int seis_step;
    int nstep_between_samples;
    type_real dt;
    type_real t0;

    Kokkos::View<type_real ***[2], Kokkos::LayoutLeft,
                 Kokkos::DefaultHostExecutionSpace>
        seismo_components;
    Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>
        h_sine_receiver_angle;
    Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>
        h_cosine_receiver_angle;
  };

public:
  SeismogramIterator() = default;

  SeismogramIterator(const int nreceivers, const int nseismograms,
                     const int max_sig_step, type_real dt, type_real t0,
                     int nstep_between_samples)
      : nreceivers(nreceivers), nseismograms(nseismograms), irec(0), iseis(0),
        nstep_between_samples(nstep_between_samples),
        max_sig_step(max_sig_step), dt(dt), t0(t0),
        h_sine_receiver_angle(
            "specfem::assembly::receivers::sine_receiver_angle", nreceivers),
        h_cosine_receiver_angle(
            "specfem::assembly::receivers::cosine_receiver_angle", nreceivers),
        seismogram_components(
            "specfem::assembly::receivers::seismogram_components", max_sig_step,
            nseismograms, nreceivers, 2),
        h_seismogram_components(
            Kokkos::create_mirror_view(seismogram_components)) {}

  Iterator begin() {
    return Iterator(irec, iseis, 0, dt, t0, nstep_between_samples,
                    h_sine_receiver_angle, h_cosine_receiver_angle,
                    h_seismogram_components);
  }

  Iterator end() {
    return Iterator(irec, iseis, max_sig_step, dt, t0, nstep_between_samples,
                    h_sine_receiver_angle, h_cosine_receiver_angle,
                    h_seismogram_components);
  }

  SeismogramIterator &get_seismogram(const std::string &station_name,
                                     const std::string &network_name,
                                     const specfem::wavefield::type type) {
    this->irec = station_network_map[station_name][network_name];
    this->iseis = seismogram_type_map[type];
    return *this;
  }

  void set_seismogram_step(const int isig_step) { this->seis_step = isig_step; }
  void set_seismogram_type(const int iseis) { this->iseis = iseis; }

  KOKKOS_FUNCTION int get_seismogram_step() const { return seis_step; }
  KOKKOS_FUNCTION int get_seis_type() const { return iseis; }

  void sync_seismograms() {
    Kokkos::deep_copy(h_seismogram_components, seismogram_components);
  }

  // Set receiver angle for a receiver
  void set_receiver_angle(int irec, type_real angle) {
    h_sine_receiver_angle(irec) = std::sin(angle);
    h_cosine_receiver_angle(irec) = std::cos(angle);
  }

private:
  int nreceivers;
  int nseismograms;
  int irec;
  int iseis;
  int nstep_between_samples;
  int max_sig_step;
  int seis_step = 0;
  type_real dt;
  type_real t0;

protected:
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>
      h_sine_receiver_angle;
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>
      h_cosine_receiver_angle;
  Kokkos::View<type_real ***[2], Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      seismogram_components;
  Kokkos::View<type_real ***[2], Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>::HostMirror
      h_seismogram_components;

  std::unordered_map<std::string, std::unordered_map<std::string, int> >
      station_network_map;
  std::unordered_map<specfem::wavefield::type, int> seismogram_type_map;
};

} // namespace specfem::assembly::receivers_impl
