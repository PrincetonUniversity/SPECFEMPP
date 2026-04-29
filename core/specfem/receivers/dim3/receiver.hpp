#pragma once

#include "specfem/constants.hpp"

#include "specfem/enums.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/setup.hpp"
#include <cmath>

namespace specfem::receivers {

template <> class receiver<specfem::element::dimension_tag::dim3> {

public:
  /**
   * Compile-time constants
   * @{
   */
  constexpr static specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim3;
  /// @}

  /**
   * @brief Construct a new receiver object
   *
   * @param network_name Name of network where this station lies in
   * @param station_name Name of station
   * @param x X coordinate of the station
   * @param y Y coordinate of the station
   * @param z Z coordinate of the station
   */
  receiver(const std::string &network_name, const std::string &station_name,
           const type_real x, const type_real y, const type_real z)
      : network_name(network_name), station_name(station_name),
        global_coordinates(x, y, z) {};

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

  int get_islice() const { return islice_; }
  void set_islice(int rank) { islice_ = rank; }

private:
  specfem::point::global_coordinates<dimension_tag>
      global_coordinates;   ///< Global coordinates of the receiver
  std::string network_name; ///< Name of the network where this station lies
  std::string station_name; ///< Name of the station
  int islice_ = -1; ///< MPI rank that owns this receiver (-1 = not yet located)
};

} // namespace specfem::receivers
