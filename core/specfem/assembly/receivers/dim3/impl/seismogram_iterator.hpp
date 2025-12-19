#pragma once

#include "enumerations/interface.hpp"
#include <array>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace specfem::assembly::receivers_impl {

//
// DIM3 SPECIALIZATION
//
template <> class SeismogramIterator<specfem::dimension::type::dim3> {
private:
  static constexpr int ncomponents = 3;

  class Iterator {
  public:
    Iterator(const int irec, const int iseis, const int seis_step,
             const type_real dt, const type_real t0,
             const int nstep_between_samples,
             Kokkos::View<type_real ***, Kokkos::DefaultHostExecutionSpace>
                 h_rotation_matrices,
             Kokkos::View<type_real ***[3], Kokkos::LayoutLeft,
                          Kokkos::DefaultHostExecutionSpace>
                 seismogram_components)
        : irec(irec), iseis(iseis), seis_step(seis_step), dt(dt), t0(t0),
          nstep_between_samples(nstep_between_samples),
          h_rotation_matrices(h_rotation_matrices),
          seismo_components(seismogram_components) {}

    std::tuple<type_real, std::array<type_real, 3> > operator*() {
      type_real time = seis_step * dt * nstep_between_samples + t0;

      std::array<type_real, 3> seismograms;

      // 3D: Use rotation matrix (nrec, 3, 3) layout
      // Apply rotation matrix: seismograms = R * raw_components
      for (int i = 0; i < 3; ++i) {
        seismograms[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
          seismograms[i] += h_rotation_matrices(irec, i, j) *
                            seismo_components(seis_step, iseis, irec, j);
        }
      }

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

    Kokkos::View<type_real ***[3], Kokkos::LayoutLeft,
                 Kokkos::DefaultHostExecutionSpace>
        seismo_components;
    Kokkos::View<type_real ***, Kokkos::DefaultHostExecutionSpace>
        h_rotation_matrices;
  };

public:
  SeismogramIterator() = default;

  SeismogramIterator(const int nreceivers, const int nseismograms,
                     const int max_sig_step, type_real dt, type_real t0,
                     int nstep_between_samples)
      : nreceivers(nreceivers), nseismograms(nseismograms), irec(0), iseis(0),
        nstep_between_samples(nstep_between_samples),
        max_sig_step(max_sig_step), dt(dt), t0(t0),
        h_rotation_matrices("specfem::assembly::receivers::rotation_matrices",
                            nreceivers, 3, 3),
        seismogram_components(
            "specfem::assembly::receivers::seismogram_components", max_sig_step,
            nseismograms, nreceivers, 3),
        h_seismogram_components(
            Kokkos::create_mirror_view(seismogram_components)) {
    // Initialize rotation matrices to identity
    for (int irec = 0; irec < nreceivers; ++irec) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          h_rotation_matrices(irec, i, j) = (i == j) ? 1.0 : 0.0;
        }
      }
    }
  }

  Iterator begin() {
    return Iterator(irec, iseis, 0, dt, t0, nstep_between_samples,
                    h_rotation_matrices, h_seismogram_components);
  }

  Iterator end() {
    return Iterator(irec, iseis, max_sig_step, dt, t0, nstep_between_samples,
                    h_rotation_matrices, h_seismogram_components);
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

  /**
   * @brief Get the time step between the samples
   *
   * @return type_real Time step between samples
   */
  type_real get_timestep() const { return dt; }

  KOKKOS_FUNCTION int get_seismogram_step() const { return seis_step; }
  KOKKOS_FUNCTION int get_seis_type() const { return iseis; }

  void sync_seismograms() {
    Kokkos::deep_copy(h_seismogram_components, seismogram_components);
  }

  // Set rotation matrix for a receiver
  void set_rotation_matrix(
      int irec,
      const std::array<std::array<type_real, 3>, 3> &rotation_matrix) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        h_rotation_matrices(irec, i, j) = rotation_matrix[i][j];
      }
    }
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
  // dim3 rotation matrix (nrec, 3, 3) - always 3x3 regardless of dimension
  Kokkos::View<type_real ***, Kokkos::DefaultHostExecutionSpace>
      h_rotation_matrices;

  Kokkos::View<type_real ***[3], Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      seismogram_components;
  Kokkos::View<type_real ***[3], Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>::HostMirror
      h_seismogram_components;

  std::unordered_map<std::string, std::unordered_map<std::string, int> >
      station_network_map;
  std::unordered_map<specfem::wavefield::type, int> seismogram_type_map;
};

} // namespace specfem::assembly::receivers_impl
