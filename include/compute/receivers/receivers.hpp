#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/element_types/element_types.hpp"
#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "receiver/interface.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <receiver/receiver.hpp>
#include <vector>

namespace specfem {
namespace compute {

namespace impl {

// Simple seismogram type iterator - just wraps a vector of seismogram types
class SeismogramTypeIterator {
public:
  using iterator = std::vector<specfem::wavefield::type>::const_iterator;

  SeismogramTypeIterator() = default;

  explicit SeismogramTypeIterator(
      const std::vector<specfem::wavefield::type> &types)
      : seismogram_types_(types) {}

  iterator begin() const { return seismogram_types_.begin(); }
  iterator end() const { return seismogram_types_.end(); }

  size_t size() const { return seismogram_types_.size(); }
  bool empty() const { return seismogram_types_.empty(); }

private:
  std::vector<specfem::wavefield::type> seismogram_types_;
};

// StationInfo that contains station data and can provide seismogram types
struct StationInfo {
  std::string network_name;
  std::string station_name;

  StationInfo(std::string network, std::string station,
              const std::vector<specfem::wavefield::type> &types)
      : network_name(std::move(network)), station_name(std::move(station)),
        seismo_types_(types) {}

  // Method to get seismogram types associated with this station
  SeismogramTypeIterator get_seismogram_types() const {
    return SeismogramTypeIterator(seismo_types_);
  }

private:
  std::vector<specfem::wavefield::type> seismo_types_;
};

// Station iterator that outputs StationInfo objects
class StationIterator {
private:
  class Iterator {
  public:
    Iterator(const StationIterator *container, size_t index)
        : container_(container), index_(index) {}

    StationInfo operator*() const {
      return StationInfo(container_->network_names_[index_],
                         container_->station_names_[index_],
                         container_->seismogram_types_);
    }

    Iterator &operator++() {
      ++index_;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return index_ != other.index_;
    }

  private:
    const StationIterator *container_;
    size_t index_;
  };

public:
  StationIterator() = default;

  StationIterator(size_t nreceivers,
                  const std::vector<specfem::wavefield::type> &seismo_types)
      : seismogram_types_(seismo_types) {
    station_names_.reserve(nreceivers);
    network_names_.reserve(nreceivers);
  }

  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, station_names_.size()); }

  size_t size() const { return station_names_.size(); }

protected:
  std::vector<std::string> station_names_;
  std::vector<std::string> network_names_;
  std::vector<specfem::wavefield::type> seismogram_types_;
};

class SeismogramIterator {
private:
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

  using ReceiverAngleType =
      Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>;
  using SeismogramType = Kokkos::View<type_real ***[2], Kokkos::LayoutLeft,
                                      Kokkos::DefaultExecutionSpace>;

public:
  SeismogramIterator() = default;

  SeismogramIterator(const int nreceivers, const int nseismograms,
                     const int max_sig_step, type_real dt, type_real t0,
                     int nstep_between_samples)
      : nreceivers(nreceivers), nseismograms(nseismograms), dt(dt), t0(t0),
        nstep_between_samples(nstep_between_samples),
        max_sig_step(max_sig_step),
        h_sine_receiver_angle(
            "specfem::compute::receivers::sine_receiver_angle", nreceivers),
        h_cosine_receiver_angle(
            "specfem::compute::receivers::cosine_receiver_angle", nreceivers),
        seismogram_components(
            "specfem::compute::receivers::seismogram_components", max_sig_step,
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

  /**
   * @brief Get the seismogram iterator object associated with a given station
   * and seismogram type
   *
   * The iterator object can be used to get the traces associated with the
   * station. See example below:
   *
   * @code
   * // Writing the x-component of the displacement seismogram in CSV
   * std::cout << "Time, value" << std::endl;
   * for (auto [time, seismograms] :
   * this->get_seismogram_iterator("STATION_NAME", "NETWORK_NAME",
   * specfem::wavefield::type::displacement)) {
   *  std::cout << time << ", " << seismograms[0] << std::endl;
   * }
   * @endcode
   * @param station_name Name of the station
   * @param network_name Name of the network
   * @param type Type of the seismogram (displacement, velocity, acceleration or
   * pressure)
   */
  SeismogramIterator &get_seismogram(const std::string &station_name,
                                     const std::string &network_name,
                                     const specfem::wavefield::type type) {
    this->irec = station_network_map[station_name][network_name];
    this->iseis = seismogram_type_map[type];
    return *this;
  }

  /** @brief Set the seismogram step
   *
   * @param isig_step Seismogram step
   */
  void set_seismogram_step(const int isig_step) { this->seis_step = isig_step; }

  /** @brief Set the seismogram type
   *
   * @param iseis Seismogram type
   */
  void set_seismogram_type(const int iseis) { this->iseis = iseis; }

  KOKKOS_FUNCTION int get_seismogram_step() const { return seis_step; }
  KOKKOS_FUNCTION int get_seis_type() const { return iseis; }

  /**
   * @brief Synchronize the seismograms from the device to the host
   *
   */
  void sync_seismograms() {
    Kokkos::deep_copy(h_seismogram_components, seismogram_components);
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
  ReceiverAngleType h_sine_receiver_angle;   ///< Sine of the receiver angle
  ReceiverAngleType h_cosine_receiver_angle; ///< Cosine of the receiver angle
  SeismogramType seismogram_components;      ///< Seismogram components
  SeismogramType::HostMirror h_seismogram_components; ///< Seismogram components
                                                      ///< stored on the host
  std::unordered_map<std::string, std::unordered_map<std::string, int> >
      station_network_map;
  std::unordered_map<specfem::wavefield::type, int> seismogram_type_map;
};
} // namespace impl

/**
 * @brief Struct to store information related to the receivers
 *
 */
struct receivers : public impl::StationIterator,
                   public impl::SeismogramIterator {
private:
  using IndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< View to store the
                                                          ///< elements
                                                          ///< associated with
                                                          ///< the receivers
  using LagrangeInterpolantType =
      Kokkos::View<type_real ***[2], Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>; ///< View to store the
                                                   ///< Lagrange interpolant for
                                                   ///< every receiver

public:
  /**
   * @brief Construct a new receivers object
   *
   */
  receivers() = default;

  /**
   * @brief Construct a new receivers object
   *
   * @param nspec Total Number of spectral elements in the domain
   * @param ngllz Total Number of GLL points in the z-direction
   * @param ngllx Total Number of GLL points in the x-direction
   * @param max_sig_step Maximum number seismogram sample points
   * @param dt Time increament
   * @param t0 Initial time
   * @param nsteps_between_samples Number of time steps between samples
   * @param receivers Vector of receivers
   * @param stypes Vector of seismogram types (displacement, velocity,
   * acceleration, pressure, or rotation)
   * @param mesh Mesh object
   * @param tags Tags for every element in the mesh
   * @param properties Properties object
   */
  receivers(const int nspec, const int ngllz, const int ngllx,
            const int max_sig_step, const type_real dt, const type_real t0,
            const int nsteps_between_samples,
            const std::vector<std::shared_ptr<specfem::receivers::receiver> >
                &receivers,
            const std::vector<specfem::wavefield::type> &stypes,
            const specfem::compute::mesh &mesh,
            const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
            const specfem::compute::element_types &element_types);

  /**
   * @brief Get the spectral element indices in which the receivers are located
   * on the device
   *
   * Returns only the indices of the elements that are associated with a
   * specific medium and property tag
   * @param medium Medium tag
   * @property property Property tag
   * @return Kokkos::View<int *, Kokkos::DefaultExecutionSpace> View of the
   * elements indices associated with the receivers
   */
  std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
  get_indices_on_device(const specfem::element::medium_tag medium,
                        const specfem::element::property_tag property) const;

  /**
   * @brief Get the spectral element indices in which the receivers are located
   * on the host
   *
   * Returns only the indices of the elements that are associated with a
   * specific medium and property tag
   * @param medium Medium tag
   * @property property Property tag
   * @return Kokkos::View<int *, Kokkos::DefaultExecutionSpace> View of the
   * elements indices associated with the receivers
   */
  std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
  get_indices_on_host(const specfem::element::medium_tag medium,
                      const specfem::element::property_tag property) const;

  /**
   * @brief Get the seismogram types
   *
   * @return std::vector<specfem::wavefield::type> Vector of seismogram types
   */
  std::vector<specfem::wavefield::type> get_seismogram_types() const {
    return seismogram_types_;
  }

  /**
   * @brief Get the station iterator
   *
   * @return const StationIterator& Iterator over stations
   */
  const impl::StationIterator &stations() const {
    return static_cast<const impl::StationIterator &>(*this);
  }

private:
  int nspec;              ///< Total number of spectral elements
  IndexViewType elements; ///< View to store the elements associated with the
                          ///< receivers
  IndexViewType::HostMirror h_elements; ///< Host view to store the
                                        ///< elements associated with the
                                        ///< receivers
  LagrangeInterpolantType lagrange_interpolant; ///< Lagrange interpolant for
                                                ///< every receiver
  LagrangeInterpolantType::HostMirror
      h_lagrange_interpolant; ///< Lagrange interpolant for every receiver
                              ///< stored on the host
  specfem::compute::element_types element_types; ///< Element types

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T),
                       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC,
                                    ISOTROPIC_COSSERAT)),
                      DECLARE((IndexViewType, receiver_indices),
                              (IndexViewType::HostMirror, h_receiver_indices),
                              (IndexViewType, elements),
                              (IndexViewType::HostMirror, h_elements)))

  template <typename ChunkIndexType, typename ViewType>
  friend KOKKOS_FUNCTION void load_on_device(const ChunkIndexType &chunk_index,
                                             const receivers &receivers,
                                             ViewType &lagrange_interpolant);

  template <typename ChunkIndexType, typename SiesmogramViewType>
  friend KOKKOS_FUNCTION void
  store_on_device(const ChunkIndexType &chunk_index,
                  const SiesmogramViewType &seismogram_components,
                  const receivers &receivers);
};

/**
 * @defgroup ComputeReceiversDataAccess
 */

/**
 * @brief Load the Lagrange interpolant for receivers associated with the
 * iterator on the device
 *
 * @ingroup ComputeReceiversDataAccess
 *
 * @tparam ChunkIndexType Chunk index type @ref
 * specfem::execution::ChunkElementIndex
 * @tparam ViewType Lagrange interpolant associated with the receivers in the
 * iterator
 *
 * @param team_member Kokkos team member
 * @param chunk_index Chunk index
 * @param receivers Receivers object containing the receiver information
 * @param lagrange_interpolant Lagrange interpolant associated with the
 * receivers in the iterator
 */
template <typename ChunkIndexType, typename ViewType>
KOKKOS_FUNCTION void load_on_device(const ChunkIndexType &chunk_index,
                                    const receivers &receivers,
                                    ViewType &lagrange_interpolant) {

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_policy_index();
        const int irec = index.imap;

#ifndef NDEBUG

        if (index.ispec >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) + ":" +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        lagrange_interpolant(ielement, index.iz, index.ix, 0) =
            receivers.lagrange_interpolant(irec, index.iz, index.ix, 0);
        lagrange_interpolant(ielement, index.iz, index.ix, 1) =
            receivers.lagrange_interpolant(irec, index.iz, index.ix, 1);
      });

  return;
}

/**
 * @brief Store the seismogram components for receivers associated with the
 * iterator on the device
 *
 * Before you store the seismogram components, you need to set the seismogram
 * time step and type. You can do this by calling the following functions:
 * @c receivers.set_seismogram_step(isig_step);
 * @c receivers.set_seismogram_type(iseis);
 *
 * @ingroup ComputeReceiversDataAccess
 * @tparam ChunkIndexType Chunk index type
 * @tparam SeismogramViewType View of the seismogram components
 * @param receivers Receivers object containing the receiver information
 */
template <typename ChunkIndexType, typename SeismogramViewType>
KOKKOS_FUNCTION void
store_on_device(const ChunkIndexType &chunk_index,
                const SeismogramViewType &seismogram_components,
                const receivers &receivers) {

  const int isig_step = receivers.get_seismogram_step();
  const int iseis = receivers.get_seis_type();

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_policy_index();
        const int irec = index.imap;

#ifndef NDEBUG

        if (index.ispec >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) + ":" +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        receivers.seismogram_components(isig_step, iseis, irec, 0) =
            seismogram_components(ielement, 0);
        receivers.seismogram_components(isig_step, iseis, irec, 1) =
            seismogram_components(ielement, 1);
      });
  return;
}

} // namespace compute
} // namespace specfem
