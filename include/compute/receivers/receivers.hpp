#pragma once

#include "compute/element_types/element_types.hpp"
#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <receiver/receiver.hpp>
#include <vector>

namespace specfem {
namespace compute {

namespace impl {
class StationIterator {
private:
  class Iterator {

  public:
    Iterator(const int irec, const int iseis, const int nreceivers,
             const int nseismograms, std::vector<std::string> &station_names,
             std::vector<std::string> &network_names,
             std::vector<specfem::wavefield::type> &seismogram_types)
        : index(irec * nseismograms + iseis), nreceivers(nreceivers),
          nseismograms(nseismograms), station_names(station_names),
          network_names(network_names), seismogram_types(seismogram_types) {}

    std::tuple<std::string, std::string, specfem::wavefield::type> operator*() {
      int irec = index / nseismograms;
      int iseis = index % nseismograms;
      return std::make_tuple(station_names[irec], network_names[irec],
                             seismogram_types[iseis]);
    }

    Iterator &operator++() {
      ++index;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return index != other.index;
    }

  private:
    int index;
    int nreceivers;
    int nseismograms;
    std::vector<std::string> &station_names;
    std::vector<std::string> &network_names;
    std::vector<specfem::wavefield::type> &seismogram_types;
  };

public:
  StationIterator() = default;

  StationIterator(const int nreceivers, const int nseismograms)
      : nreceivers(nreceivers), nseismograms(nseismograms),
        station_names(nreceivers), network_names(nreceivers),
        seismogram_types(nseismograms) {}

  Iterator begin() {
    return Iterator(0, 0, nreceivers, nseismograms, station_names,
                    network_names, seismogram_types);
  }

  Iterator end() {
    return Iterator(nreceivers - 1, nseismograms, nreceivers, nseismograms,
                    station_names, network_names, seismogram_types);
  }

  /** @brief Get the station iterator object
   *
   * The iterator object can be used to iterate over the stations. See example
   * below:
   * @code
   * // Iterator over all the stations
   * for (auto [station_name, network_name, type] :
   * this->get_stations()) {
   * // Use get_seismogram(station_name, network_name, type) to get the traces
   * }
   * @endcode
   * @return StationIterator Iterator over the stations
   */
  StationIterator &get_stations() { return *this; }

private:
  int nreceivers;
  int nseismograms;

protected:
  std::vector<std::string> station_names;                 ///< Station names
  std::vector<std::string> network_names;                 ///< Network names
  std::vector<specfem::wavefield::type> seismogram_types; ///< Seismogram types
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

  SeismogramIterator(const int nreceivers, const int nsiesmograms,
                     const int max_sig_step, type_real dt, type_real t0,
                     int nstep_between_samples)
      : nreceivers(nreceivers), nsiesmograms(nsiesmograms), dt(dt), t0(t0),
        nstep_between_samples(nstep_between_samples),
        max_sig_step(max_sig_step),
        h_sine_receiver_angle(
            "specfem::compute::receivers::sine_receiver_angle", nreceivers),
        h_cosine_receiver_angle(
            "specfem::compute::receivers::cosine_receiver_angle", nreceivers),
        seismogram_components(
            "specfem::compute::receivers::seismogram_components", max_sig_step,
            nsiesmograms, nreceivers, 2),
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
  int nsiesmograms;
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
   * acceleration or pressure)
   * @param mesh Mesh object
   * @param tags Tags for every element in the mesh
   * @param properties Properties object
   */
  receivers(const int nspec, const int ngllz, const int ngllx,
            const int max_sig_step, const type_real dt, const type_real t0,
            const int nsteps_between_samples,
            const std::vector<std::shared_ptr<specfem::receivers::receiver> >
                &receivers,
            const std::vector<specfem::enums::seismogram::type> &stypes,
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
    return seismogram_types;
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

#define RECEIVER_INDICES_VARIABLE_NAME(DIMENSION_TAG, MEDIUM_TAG,              \
                                       PROPERTY_TAG)                           \
  IndexViewType CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENSION_TAG),        \
                                     GET_NAME(MEDIUM_TAG),                     \
                                     GET_NAME(PROPERTY_TAG));                  \
  IndexViewType::HostMirror CREATE_VARIABLE_NAME(                              \
      h_elements, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),               \
      GET_NAME(PROPERTY_TAG));                                                 \
  IndexViewType CREATE_VARIABLE_NAME(                                          \
      receiver_indices, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),         \
      GET_NAME(PROPERTY_TAG));                                                 \
  IndexViewType::HostMirror CREATE_VARIABLE_NAME(                              \
      h_receiver_indices, GET_NAME(DIMENSION_TAG), GET_NAME(MEDIUM_TAG),       \
      GET_NAME(PROPERTY_TAG));

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(RECEIVER_INDICES_VARIABLE_NAME,
                                      WHERE(DIMENSION_TAG_DIM2)
                                          WHERE(MEDIUM_TAG_ELASTIC_SV,
                                                MEDIUM_TAG_ACOUSTIC)
                                              WHERE(PROPERTY_TAG_ISOTROPIC,
                                                    PROPERTY_TAG_ANISOTROPIC))

#undef RECEIVER_INDICES_VARIABLE_NAME

  template <typename MemberType, typename IteratorType, typename ViewType>
  friend KOKKOS_FUNCTION void
  load_on_device(const MemberType &team_member, const IteratorType &iterator,
                 const receivers &receivers, ViewType &lagrange_interpolant);

  template <typename MemberType, typename IteratorType,
            typename SiesmogramViewType>
  friend KOKKOS_FUNCTION void
  store_on_device(const MemberType &team_member, const IteratorType &iterator,
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
 * @tparam MemberType Kokkos team member type
 * @tparam IteratorType Chunk policy iterator type @ref
 * specfem::policy::element_chunk
 * @param receivers Receivers object containing the receiver information
 * @tparam ViewType Lagrange interpolant associated with the receivers in the
 * iterator
 */
template <typename MemberType, typename IteratorType, typename ViewType>
KOKKOS_FUNCTION void
load_on_device(const MemberType &team_member, const IteratorType &iterator,
               const receivers &receivers, ViewType &lagrange_interpolant) {

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, iterator.chunk_size()),
      [&](const int i) {
        const auto iterator_index = iterator(i);
        const auto index = iterator_index.index;

#ifndef NDEBUG

        if (index.ispec >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) + ":" +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        const int irec = iterator_index.imap;

        lagrange_interpolant(iterator_index.ielement, index.iz, index.ix, 0) =
            receivers.lagrange_interpolant(irec, index.iz, index.ix, 0);
        lagrange_interpolant(iterator_index.ielement, index.iz, index.ix, 1) =
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
 * @tparam MemberType Kokkos team member type
 * @tparam IteratorType Chunk policy iterator type @ref
 * specfem::policy::element_chunk
 * @tparam SeismogramViewType View of the seismogram components
 * @param receivers Receivers object containing the receiver information
 */
template <typename MemberType, typename IteratorType,
          typename SeismogramViewType>
KOKKOS_FUNCTION void
store_on_device(const MemberType &team_member, const IteratorType &iterator,
                const SeismogramViewType &seismogram_components,
                const receivers &receivers) {

  const int isig_step = receivers.get_seismogram_step();
  const int iseis = receivers.get_seis_type();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, iterator.chunk_size()),
      [&](const int i) {
        const auto iterator_index = iterator(i);
        const auto index = iterator_index.index;

#ifndef NDEBUG

        if (index.ispec >= receivers.nspec) {
          std::string message = "Invalid element detected in kernel at " +
                                std::string(__FILE__) +
                                std::to_string(__LINE__);
          Kokkos::abort(message.c_str());
        }

#endif

        const int irec = iterator_index.imap;

        receivers.seismogram_components(isig_step, iseis, irec, 0) =
            seismogram_components(iterator_index.ielement, 0);

        receivers.seismogram_components(isig_step, iseis, irec, 1) =
            seismogram_components(iterator_index.ielement, 1);
      });

  return;
}

} // namespace compute
} // namespace specfem
