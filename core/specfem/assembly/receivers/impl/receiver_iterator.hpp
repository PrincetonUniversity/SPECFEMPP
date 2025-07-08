#include "enumerations/interface.hpp"
#include <vector>

namespace specfem::assembly::receivers_impl {

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

} // namespace specfem::assembly::receivers_impl
