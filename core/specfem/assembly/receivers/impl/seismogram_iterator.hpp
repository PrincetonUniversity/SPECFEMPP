#pragma once

#include "specfem/assembly/receivers/impl/rotate_components.hpp"
#include "specfem/assembly/receivers/impl/seismogram_types.hpp"
#include "specfem/element/dimension.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/enums.hpp"
#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace specfem::assembly::receivers_impl {

/**
 * @brief Unified seismogram iterator for 2D and 3D simulations.
 *
 * Both dimensions use a rotation matrix (nrec x ncomponents x ncomponents)
 * mapping Cartesian mesh components (columns) to output seismogram components
 * (rows). It is applied on the device at sampling time by @c store_on_device,
 * so the stored components, and what the writer reads through this iterator,
 * are already rotated.
 *
 * - dim2: 2x2 rotation matrix built from the receiver angle (cos/sin)
 * - dim3: 3x3 rotation matrix set explicitly per receiver
 *
 * @tparam DimensionTag specfem::element::dimension_tag::dim2 or dim3
 */
template <specfem::element::dimension_tag DimensionTag>
class SeismogramIterator {
private:
  static constexpr int ncomponents =
      specfem::element::dimension<DimensionTag>::dim;

  class Iterator {
  public:
    Iterator(const int irec, const int iseis, const int seis_step,
             const type_real dt, const type_real t0,
             const int nstep_between_samples,
             Kokkos::View<type_real ***[ncomponents], Kokkos::LayoutLeft,
                          Kokkos::DefaultHostExecutionSpace>
                 seismogram_components)
        : irec(irec), iseis(iseis), seis_step(seis_step), dt(dt), t0(t0),
          nstep_between_samples(nstep_between_samples),
          seismo_components(seismogram_components) {}

    std::tuple<type_real, std::array<type_real, ncomponents>> operator*() {
      type_real time = seis_step * dt * nstep_between_samples + t0;

      std::array<type_real, ncomponents> seismograms;

      for (int icomp = 0; icomp < ncomponents; ++icomp) {
        seismograms[icomp] = seismo_components(seis_step, iseis, irec, icomp);
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

    Kokkos::View<type_real ***[ncomponents], Kokkos::LayoutLeft,
                 Kokkos::DefaultHostExecutionSpace>
        seismo_components;
  };

public:
  SeismogramIterator() = default;

  SeismogramIterator(const int nreceivers,
                     const SeismogramTypes &seismogram_types,
                     const int max_sig_step, type_real dt, type_real t0,
                     int nstep_between_samples)
      : nreceivers(nreceivers), seismogram_type_set_(seismogram_types), irec(0),
        iseis(0), nstep_between_samples(nstep_between_samples),
        max_sig_step(max_sig_step), dt(dt), t0(t0),
        rotation_matrices("specfem::assembly::receivers::rotation_matrices",
                          nreceivers, ncomponents, ncomponents),
        h_rotation_matrices(Kokkos::create_mirror_view(rotation_matrices)),
        seismogram_components(
            "specfem::assembly::receivers::seismogram_components", max_sig_step,
            seismogram_type_set_.size(), nreceivers, ncomponents),
        h_seismogram_components(
            Kokkos::create_mirror_view(seismogram_components)) {
    // Initialize rotation matrices to identity
    for (int irec_local = 0; irec_local < nreceivers; ++irec_local) {
      for (int i = 0; i < ncomponents; ++i) {
        for (int j = 0; j < ncomponents; ++j) {
          h_rotation_matrices(irec_local, i, j) = (i == j) ? 1.0 : 0.0;
        }
      }
    }
    Kokkos::deep_copy(rotation_matrices, h_rotation_matrices);
  }

  Iterator begin() {
    return Iterator(irec, iseis, 0, dt, t0, nstep_between_samples,
                    h_seismogram_components);
  }

  Iterator end() {
    return Iterator(irec, iseis, max_sig_step, dt, t0, nstep_between_samples,
                    h_seismogram_components);
  }

  SeismogramIterator &get_seismogram(const std::string &station_name,
                                     const std::string &network_name,
                                     const specfem::enums::wavefield type) {
    this->irec = station_network_map[station_name][network_name];
    this->iseis = seismogram_type_set_.index_of(type);
    return *this;
  }

  void set_seismogram_step(const int isig_step) { this->seis_step = isig_step; }
  void set_seismogram_type(const int iseis) { this->iseis = iseis; }

  KOKKOS_FUNCTION int get_seismogram_step() const { return seis_step; }
  KOKKOS_FUNCTION int get_seis_type() const { return iseis; }

  type_real get_t0() const { return t0; }
  int get_nsteps() const { return max_sig_step; }

  void sync_seismograms() {
    Kokkos::deep_copy(h_seismogram_components, seismogram_components);
  }

  void gather_to_main() {
#ifdef SPECFEM_ENABLE_MPI
    const int count = static_cast<int>(h_seismogram_components.span());
    if (specfem::MPI::main_proc()) {
      SPECFEM_MPI_SAFECALL(MPI_Reduce(
          MPI_IN_PLACE, h_seismogram_components.data(), count,
          SPECFEM_MPI_TYPE_REAL, MPI_SUM, 0, specfem::MPI::communicator()));
    } else {
      SPECFEM_MPI_SAFECALL(MPI_Reduce(h_seismogram_components.data(), nullptr,
                                      count, SPECFEM_MPI_TYPE_REAL, MPI_SUM, 0,
                                      specfem::MPI::communicator()));
    }
#endif
  }

  /**
   * @brief Set the rotation matrix of a receiver on the host
   *
   * Rows are output components, columns are mesh components. Takes effect on
   * the device after sync_rotation_matrices().
   *
   * @tparam ElementType Element type of the input matrix (narrowed to
   * type_real)
   * @param irec_local Local receiver index
   * @param rotation_matrix ncomponents x ncomponents rotation matrix
   */
  template <typename ElementType>
  void
  set_rotation_matrix(const int irec_local,
                      const std::array<std::array<ElementType, ncomponents>,
                                       ncomponents> &rotation_matrix) {
    for (int i = 0; i < ncomponents; ++i) {
      for (int j = 0; j < ncomponents; ++j) {
        h_rotation_matrices(irec_local, i, j) =
            static_cast<type_real>(rotation_matrix[i][j]);
      }
    }
  }

  /**
   * @brief Get the host rotation matrix of a receiver
   *
   * @param irec_local Local receiver index
   * @return ncomponents x ncomponents matrix (rows: output components)
   */
  std::array<std::array<type_real, ncomponents>, ncomponents>
  get_rotation_matrix(const int irec_local) const {
    std::array<std::array<type_real, ncomponents>, ncomponents> matrix;
    for (int i = 0; i < ncomponents; ++i) {
      for (int j = 0; j < ncomponents; ++j) {
        matrix[i][j] = h_rotation_matrices(irec_local, i, j);
      }
    }
    return matrix;
  }

  /// Copy the host rotation matrices to the device.
  void sync_rotation_matrices() {
    Kokkos::deep_copy(rotation_matrices, h_rotation_matrices);
  }

private:
  int nreceivers;
  SeismogramTypes seismogram_type_set_;
  int irec;
  int iseis;
  int nstep_between_samples;
  int max_sig_step;
  int seis_step = 0;
  type_real dt;
  type_real t0;

protected:
  Kokkos::View<type_real ***, Kokkos::DefaultExecutionSpace>
      rotation_matrices; ///< (nrec, ncomp, ncomp) rotation applied on the
                         ///< device at sampling time
  Kokkos::View<type_real ***, Kokkos::DefaultExecutionSpace>::host_mirror_type
      h_rotation_matrices;

  Kokkos::View<type_real ***[ncomponents], Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      seismogram_components;
  Kokkos::View<type_real ***[ncomponents], Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>::host_mirror_type
      h_seismogram_components;

  std::unordered_map<std::string, std::unordered_map<std::string, int>>
      station_network_map;
};

} // namespace specfem::assembly::receivers_impl
