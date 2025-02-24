#pragma once
#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store coordinates for a 3D mesh
 *
 */
template <> struct coordinates<specfem::dimension::type::dim3> {
  constexpr static auto dimension = specfem::dimension::type::dim3;

  using UniqueView = Kokkos::View<type_real *, Kokkos::HostSpace>;
  using LocalView = Kokkos::View<type_real ****, Kokkos::HostSpace>;

  // Parameters
  int nspec;
  int nglob;
  int ngllx;
  int nglly;
  int ngllz;

  UniqueView x;
  UniqueView y;
  UniqueView z;

  /**
   * @name Constructors
   *
   * Constructors for the coordinates struct
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  coordinates(){}; // Default constructor

  /**
   * @brief Constructor
   *
   * Constructor for the coordinates struct that initializes the number of
   * the underlying arrays with the given parameters
   *
   * @param nspec Total number of spectral elements
   * @param nglob Total number of global nodes
   * @param ngllx Number of GLL points in the x direction
   * @param nglly Number of GLL points in the y direction
   * @param ngllz Number of GLL points in the z direction
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * specfem::mesh::coordinates<specfem::dimension::type::dim3> coordinates(
   *   nspec, nglob, ngllx, nglly, ngllz);
   *
   * // Read coordinates from file
   * specfem::IO::mesh::impl::fortran::dim3::read_array(stream, coordinates.x);
   * specfem::IO::mesh::impl::fortran::dim3::read_array(stream, coordinates.y);
   * specfem::IO::mesh::impl::fortran::dim3::read_array(stream, coordinates.z);
   * @endcode
   *
   */
  coordinates(int nspec, int nglob, int ngllx, int nglly, int ngllz)
      : nspec(nspec), nglob(nglob), ngllx(ngllx), nglly(nglly), ngllz(ngllz),
        x("x", nglob), y("y", nglob), z("z", nglob){};

  ///@} // Constructors
  /**
   * @brief Print the coordinate specs
   *
   * @code{.cpp}
   * // Example of how to use this function
   * coordinates.print();
   * @endcode
   */
  void print() const;

  /**
   * @brief Print the coordinates at a specific global node
   *
   * @code
   * // Example of how to use this function
   * int iglob = 10;
   * coordinates.print(iglob);
   * @endcode
   */
  void print(int iglob) const;
};

} // namespace mesh
} // namespace specfem
