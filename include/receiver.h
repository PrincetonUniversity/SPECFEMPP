#ifndef RECEIVER_H
#define RECEIVER_H

#include "../include/constants.h"
#include "../include/enums.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/specfem_mpi.h"
#include "../include/specfem_setup.hpp"
#include <cmath>

namespace specfem {
namespace receivers {

/**
 * @brief Receiver Class
 *
 */
class receiver {

public:
  /**
   * @brief Construct a new receiver object
   *
   * @param network_name Name of network where this station lies in
   * @param station_name Name of station
   * @param x X coordinate of the station
   * @param z Z coordinate of the station
   * @param angle Angle of the station
   */
  receiver(const std::string network_name, const std::string station_name,
           const type_real x, const type_real z, const type_real angle)
      : network_name(network_name), station_name(station_name), x(x), z(z),
        angle(angle){};
  /**
   * @brief Locate station within the mesh
   *
   * Given the global cartesian coordinates of a station, locate the spectral
   * element and xi, gamma value of the station
   *
   * @param h_ibool Global number for every quadrature point
   * @param coord (x, z) for every distinct control node
   * @param xigll Quadrature points in x-dimension
   * @param zigll Quadrature points in z-dimension
   * @param nproc Number of processors in the simulation
   * @param coorg Value of every spectral element control nodes
   * @param knods Global control element number for every control node
   * @param npgeo Total number of distinct control nodes
   * @param ispec_type material type for every spectral element
   * @param mpi Pointer to specfem MPI object
   */
  void locate(
      const specfem::kokkos::HostView2d<type_real> coord,
      const specfem::kokkos::HostMirror3d<int> h_ibool,
      const specfem::kokkos::HostMirror1d<type_real> xigll,
      const specfem::kokkos::HostMirror1d<type_real> zigll, const int nproc,
      const specfem::kokkos::HostView2d<type_real> coorg,
      const specfem::kokkos::HostView2d<int> knods, const int npgeo,
      const specfem::kokkos::HostMirror1d<specfem::elements::type> ispec_type,
      const specfem::MPI::MPI *mpi);
  /**
   * @brief Compute the receiver array (lagrangians) for this station
   *
   * @param quadx Quadrature object in x-dimension
   * @param quadz Quadrature object in z-dimension
   * @param receiver_array view to store the source array
   */
  void
  compute_receiver_array(const specfem::quadrature::quadrature &quadx,
                         const specfem::quadrature::quadrature &quadz,
                         specfem::kokkos::HostView3d<type_real> receiver_array);
  /**
   * @brief Check if the station is within the domain
   *
   * @param xmin minimum x-coordinate on my processor
   * @param xmax maximum x-coordinate on my processor
   * @param zmin minimum z-coordinate on my processor
   * @param zmax maximum z-coordinate on my processor
   * @param mpi Pointer to specfem MPI object
   */
  void check_locations(const type_real xmin, const type_real xmax,
                       const type_real zmin, const type_real zmax,
                       const specfem::MPI::MPI *mpi);
  /**
   * @brief Get the MPI slice where this station lies
   *
   * @return int MPI slice where this station lies
   */
  int get_islice() { return this->islice; };
  /**
   * @brief Get the spectral element where this station lies
   *
   * @return int Spectral element where this station lies
   */
  int get_ispec() { return this->ispec; };
  /**
   * @brief Get the sine of angle of this station
   *
   * @return type_real Sine value of the angle of this station
   */
  type_real get_sine() {
    return std::sin(Kokkos::numbers::pi_v<type_real> / 180 * this->angle);
  }
  /**
   * @brief Get the cosine of angle of this station
   *
   * @return type_real Coine value of the angle of this station
   */
  type_real get_cosine() {
    return std::cos(Kokkos::numbers::pi_v<type_real> / 180 * this->angle);
  }
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

private:
  type_real xi;    ///< f$ \xi f$ value of source inside element
  type_real gamma; ///< f$ \gamma f$ value of source inside element
  type_real x;     ///< x coordinate of source
  type_real z;     ///< z coordinate of source
  int ispec;       ///< ispec element number where source is located
  int islice;      ///< MPI slice (rank) where the source is located
  specfem::elements::type el_type; ///< type of the element inside which this
                                   ///< receiver lies
  type_real angle;                 ///< Angle to rotate components at receivers
  std::string network_name; ///< Name of the network where this station lies
  std::string station_name; ///< Name of the station
};
} // namespace receivers

} // namespace specfem

#endif
