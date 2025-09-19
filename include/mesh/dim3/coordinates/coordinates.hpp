#pragma once
#include "enumerations/dimension.hpp"
#include "mesh/dim3/mapping/mapping.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"

#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>

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
  coordinates() {}; // Default constructor

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
   * specfem::io::mesh::impl::fortran::dim3::read_array(stream, coordinates.x);
   * specfem::io::mesh::impl::fortran::dim3::read_array(stream, coordinates.y);
   * specfem::io::mesh::impl::fortran::dim3::read_array(stream, coordinates.z);
   * @endcode
   *
   */
  coordinates(int nspec, int nglob, int ngllz, int nglly, int ngllx)
      : nspec(nspec), nglob(nglob), ngllx(ngllx), nglly(nglly), ngllz(ngllz),
        x("x", nglob), y("y", nglob), z("z", nglob) {};

  ///@} // Constructors
  /**
   * @brief Print the coordinate specs
   *
   * @code{.cpp}
   * // Example of how to use this function
   * coordinates.print();
   * @endcode
   */
  std::string print() const;

  /**
   * @brief Print the coordinates at a specific global node
   *
   * @code
   * // Example of how to use this function
   * int iglob = 10;
   * coordinates.print(iglob);
   * @endcode
   */
  std::string print(int iglob) const;

  /**
   * @brief Print the coordinates at a specific spectral element
   *
   * @param ispec Spectral element number
   *
   * @code
   * // Example of how to use this function
   * int ispec = 10;
   * coordinates.print(ispec);
   * @endcode
   */
  std::string print(int ispec, specfem::mesh::mapping<dimension> &mapping,
                    const std::string component) const;

  /**
   * @brief Compute bounding box of the mesh`
   *
   * @return std::array<type_real, 6> Array containing minima and maxima of
   *         domain @code {xmin, xmax, ymin, ymax, zmin, zmax} @endcode
   */
  std::array<type_real, 6> bounding_box() const;
};

} // namespace mesh
} // namespace specfem
