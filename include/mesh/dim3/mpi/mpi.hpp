#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store communication information between different MPI slices
 *
 * @tparam DimensionTag Dimension type
 */
template <> struct mpi<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

  int num_interfaces;        ///< Number of MPI interfaces
  int max_nibool_interfaces; ///< Maximum number of GLL point interfaces

  Kokkos::View<int *, Kokkos::HostSpace> neighbors; ///< Neighbor MPI ranks for
                                                    ///< this MPI slice
  Kokkos::View<int *, Kokkos::HostSpace> nibool_interfaces; ///<
  Kokkos::View<int **, Kokkos::HostSpace> ibool_interfaces; ///< My neighbor MPI
                                                            ///< interfaces

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  mpi() {};

  /**
   * @brief Constructor
   *
   * Constructor for the mpi_interfaces struct that initializes the number of
   * the underlying arrays with the given parameters
   *
   * @param num_interfaces Number of MPI interfaces
   * @param max_nibool_interfaces Maximum number of GLL point interfaces
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * specfem::mesh::mpi_interfaces<specfem::dimension::type::dim3>
   * mpi_interfaces( num_interfaces, max_nibool_interfaces);
   * @endcode
   *
   */
  mpi(const int num_interfaces, const int max_nibool_interfaces)
      : num_interfaces(num_interfaces),
        max_nibool_interfaces(max_nibool_interfaces),
        neighbors("neighbors", num_interfaces),
        nibool_interfaces("nibool_interfaces", num_interfaces),
        ibool_interfaces("ibool_interfaces", num_interfaces,
                         max_nibool_interfaces) {};
};

} // namespace mesh
} // namespace specfem
