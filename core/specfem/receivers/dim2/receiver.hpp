#pragma once

#include "specfem/constants.hpp"
#include "specfem/coordinate_systems/coordinate_resolution_result.hpp"
#include "specfem/coordinate_systems/coordinates.hpp"

#include "specfem/enums.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/setup.hpp"
#include <cmath>
#include <optional>

namespace specfem::receivers {

template <> class receiver<specfem::element::dimension_tag::dim2> {

public:
  /**
   * Compile-time constants
   * @{
   */
  constexpr static specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim2;
  /// @}

  /**
   * @brief Construct a new receiver object
   *
   * @param network_name Name of network where this station lies in
   * @param station_name Name of station
   * @param x X coordinate of the station
   * @param z Z coordinate of the station
   * @param angle Angle of the station
   */
  receiver(const std::string &network_name, const std::string &station_name,
           const type_real x, const type_real z, const type_real angle)
      : network_name(network_name), station_name(station_name),
        global_coordinates(x, z), angle(angle) {};

  /**
   * @brief Get the name of network where this station lies
   *
   * @return std::string name of the network where the station lies
   */
  std::string get_network_name() { return this->network_name; }
  /**
   * @brief Get the name of this station
   *
   * @return std::string Name of this station
   */
  std::string get_station_name() { return this->station_name; }

  /**
   * @brief User output
   *
   */
  std::string print() const;

  type_real get_angle() const { return this->angle; }

  /**
   * @brief Get the global coordinates of the receiver
   *
   * @return specfem::point::global_coordinates<dimension_tag>
   */
  specfem::point::global_coordinates<dimension_tag>
  get_global_coordinates() const {
    return global_coordinates;
  }

  /**
   * @brief Set the global coordinates of the receiver
   *
   * @param global_coordinates global coordinates
   */
  void
  set_global_coordinates(const specfem::point::global_coordinates<dimension_tag>
                             &global_coordinates) {
    this->global_coordinates = global_coordinates;
  }

  /**
   * @brief Equality operator
   *
   * @param other Other receiver to compare with
   * @return true if receivers are equal, false otherwise
   */
  bool operator==(const receiver &other) const;

  int get_partition_index() const { return partition_index_; }
  void set_partition_index(int rank) { partition_index_ = rank; }

  /**
   * @brief Set the coordinate resolution result (resolved global + topography).
   *
   * dim2 receivers are specified directly, so this is normally left unset; the
   * accessor exists for a uniform interface with dim3.
   */
  void set_resolution_result(
      const specfem::coordinate_systems::CoordinateResolutionResult<
          dimension_tag> &resolution) {
    resolution_ = resolution;
  }

  /**
   * @brief Get the coordinate resolution result, or nullopt if not resolved.
   */
  const std::optional<
      specfem::coordinate_systems::CoordinateResolutionResult<dimension_tag>> &
  get_resolution_result() const {
    return resolution_;
  }

  /**
   * @brief Set the location error (target-to-found distance) in metres.
   */
  void set_location_error(type_real error) { location_error_ = error; }

  /**
   * @brief Get the location error (target-to-found distance) in metres.
   */
  type_real get_location_error() const { return location_error_; }

  /**
   * @brief Set the generic coordinates for this receiver.
   */
  void set_read_coordinates(
      std::unique_ptr<specfem::coordinate_systems::coordinates<dimension_tag>>
          coordinates) {
    read_coordinates_ = std::move(coordinates);
  }

  /**
   * @brief Get the generic coordinates (const), or nullptr if not set.
   */
  const specfem::coordinate_systems::coordinates<dimension_tag> *
  get_read_coordinates() const {
    return read_coordinates_.get();
  }

  /**
   * @brief Get the generic coordinates (mutable), or nullptr if not set.
   */
  specfem::coordinate_systems::coordinates<dimension_tag> *
  get_read_coordinates() {
    return read_coordinates_.get();
  }

private:
  specfem::point::global_coordinates<dimension_tag>
      global_coordinates; ///< Global coordinates of the receiver
  std::unique_ptr<specfem::coordinate_systems::coordinates<dimension_tag>>
      read_coordinates_; ///< Generic coordinates (resolved at assembly time)
  std::optional<
      specfem::coordinate_systems::CoordinateResolutionResult<dimension_tag>>
      resolution_;          ///< Resolved global + topography (generic coords)
  type_real angle;          ///< Angle to rotate components at receivers
  std::string network_name; ///< Name of the network where this station lies
  std::string station_name; ///< Name of the station
  int partition_index_ =
      -1; ///< MPI rank that owns this receiver (-1 = not yet located)
  type_real location_error_ =
      -1; ///< Target-to-found distance in metres (-1 = not yet located)
};

} // namespace specfem::receivers
