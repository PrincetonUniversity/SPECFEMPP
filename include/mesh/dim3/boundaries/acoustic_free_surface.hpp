#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

template <> struct acoustic_free_surface<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type
  // const int ndim = 3; ///< Dimension

  int num_free_surface_faces; ///< Number of boundaries faces
  int nelements;              ///< Number of elements on the boundary
  int ngllsquare;             ///< Number of GLL points squared

  Kokkos::View<int *, Kokkos::HostSpace> ispec; ///< Spectral element index for
                                                ///< elements on the boundary
  Kokkos::View<int ***, Kokkos::HostSpace> ijk; ///< Which edge of the element
                                                ///< is on the boundary
  Kokkos::View<type_real **, Kokkos::HostSpace> jacobian2Dw; ///< Jacobian of
                                                             ///< the 2D
  Kokkos::View<type_real ***, Kokkos::HostSpace> normal; ///< Jacobian of the 2D

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  acoustic_free_surface() {};

  /**
   * @brief Constructor
   *
   * Constructor for the acoustic_free_surface struct that initializes the
   * number of the underlying arrays with the given parameters
   *
   * @param num_free_surface_faces Number of free surface faces
   * @param ngllsquare Number of GLL points squared
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * specfem::mesh::acoustic_free_surface<specfem::dimension::type::dim3>
   * acoustic_free_surface( num_free_surface_faces, ngllsquare);
   * @endcode
   *
   */
  acoustic_free_surface(const int num_free_surface_faces, const int ngllsquare)
      : nelements(num_free_surface_faces), ngllsquare(ngllsquare),
        num_free_surface_faces(num_free_surface_faces) {

    ispec = Kokkos::View<int *, Kokkos::HostSpace>("ispec", nelements);
    ijk = Kokkos::View<int ***, Kokkos::HostSpace>("ijk", nelements, 3,
                                                   ngllsquare);
    jacobian2Dw = Kokkos::View<type_real **, Kokkos::HostSpace>(
        "jacobian2Dw", nelements, ngllsquare);
    // ndim=3

    normal = Kokkos::View<type_real ***, Kokkos::HostSpace>("normal", nelements,
                                                            3, ngllsquare);
  }
  ///@}

  /**
   * @brief Print the acoustic_free_surface struct
   *
   * Print the acoustic_free_surface struct to the console
   *
   * @code{.cpp}
   * // Example of how to use this function
   * acoustic_free_surface.print();
   * @endcode
   *
   */
  std::string print() const;

  /**
   * @brief Print the ijk values for a given face
   *
   * Print the ijk values for a given face
   *
   * @param iface index of the face
   *
   * @code{.cpp}
   * // Example of how to use this function
   * acoustic_free_surface.print_ijk(iface);
   * @endcode
   *
   */
  std::string print_ijk(const int iface) const;
};

} // namespace mesh
} // namespace specfem
