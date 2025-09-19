#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store absorbing boundaries
 *
 */
template <specfem::element::medium_tag MediumTag> struct stacey_mass {};

/**
 * @brief Struct to store absorbing boundaries for an elastic medium
 *
 */
template <> struct stacey_mass<specfem::element::medium_tag::elastic> {
  int nglob;             ///< Number of GLL points
  bool acoustic = false; ///< Flag for acoustic simulation
  bool elastic = true;   ///< Flag for elastic simulation

  Kokkos::View<type_real *, Kokkos::HostSpace> x;
  Kokkos::View<type_real *, Kokkos::HostSpace> y;
  Kokkos::View<type_real *, Kokkos::HostSpace> z;

  /** @name Stacey Elastic Mass struct constructors
   *  @{
   */
  /**
   * @brief Default constructor
   *
   */
  stacey_mass() {};

  /**
   * @brief Constructor
   *
   * @param nglob Number of GLL points
   *
   * @note The x, y, and z views are created
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * const int nglob = 10;
   * stacey_mass<specfem::element::medium_tag::elastic> mass_matrix(nglob);
   *
   * // Populate the mass matrix from the binary file
   * specfem::io::mesh::impl::fortran::dim3::read_array(stream, mass_matrix.x);
   * specfem::io::mesh::impl::fortran::dim3::read_array(stream, mass_matrix.y);
   * specfem::io::mesh::impl::fortran::dim3::read_array(stream, mass_matrix.z);
   * @endcode
   */
  stacey_mass(const int nglob) : nglob(nglob) {
    x = Kokkos::View<type_real *, Kokkos::HostSpace>("rmass_x", nglob);
    y = Kokkos::View<type_real *, Kokkos::HostSpace>("rmass_y", nglob);
    z = Kokkos::View<type_real *, Kokkos::HostSpace>("rmass_z", nglob);
  }
  /** @} */ // Stacey Elastic Mass struct constructors
};

/**
 * @brief Struct to store absorbing boundaries for an acoustic medium
 *
 */
template <> struct stacey_mass<specfem::element::medium_tag::acoustic> {
  int nglob;            ///< Number of GLL points
  bool acoustic = true; ///< Flag for acoustic simulation
  bool elastic = false; ///< Flag for elastic simulation

  Kokkos::View<type_real *, Kokkos::HostSpace> mass;

  /** @name Constructors
   *  @{
   */

  /**
   * @brief Default constructor
   *
   */
  stacey_mass() {};

  /**
   * @brief Constructor
   *
   * @param nglob Number of GLL points
   *
   * @note The mass view is created with the name "rmass_x"
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * const int nglob = 10;
   * stacey_mass<specfem::element::medium_tag::acoustic> mass_matrix(nglob);
   *
   * // Populate the mass matrix from the binary file
   * specfem::io::mesh::impl::fortran::dim3::read_array(stream,
   * mass_matrix.mass);
   * @endcode
   */
  stacey_mass(const int nglob) : nglob(nglob) {
    mass = Kokkos::View<type_real *, Kokkos::HostSpace>("rmass_x", nglob);
  }
  /** @} */ // Constructors
};

/**
 * @brief Struct to store absorbing boundaries for 3D meshes
 */
template <> struct absorbing_boundary<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type
  // const int ndim = 3; ///< Dimension

  int num_abs_boundary_faces; ///< Number of boundaries faces
  int ngllsquare;             ///< Number of GLL points squared
  bool acoustic = false;      ///< Flag for acoustic simulation
  bool elastic = false;       ///< Flag for elastic simulation

  int nspec2D_xmin, nspec2D_xmax; /// Number of elements on the boundaries
  int nspec2D_ymin, nspec2D_ymax;
  int NSPEC2D_BOTTOM, NSPEC2D_TOP;

  Kokkos::View<int *, Kokkos::HostSpace> ibelm_xmin,
      ibelm_xmax; ///< Spectral
                  ///< element
                  ///< index for
                  ///< elements on
                  ///< the boundary
  Kokkos::View<int *, Kokkos::HostSpace> ibelm_ymin, ibelm_ymax;
  Kokkos::View<int *, Kokkos::HostSpace> ibelm_bottom, ibelm_top;

  Kokkos::View<int *, Kokkos::HostSpace> ispec; ///< Spectral element index for
                                                ///< elements on the boundary
  Kokkos::View<int ***, Kokkos::HostSpace> ijk; ///< Which edge of the element
                                                ///< is on the boundary
  Kokkos::View<type_real **, Kokkos::HostSpace> jacobian2Dw; ///< Jacobian of
                                                             ///< the 2D
  Kokkos::View<type_real ***, Kokkos::HostSpace> normal; ///< Jacobian of the 2D

  stacey_mass<specfem::element::medium_tag::elastic> mass_elastic;
  stacey_mass<specfem::element::medium_tag::acoustic> mass_acoustic;

  /** @name Constructors
   *  @{
   */

  /**
   * @brief Default constructor
   *
   */
  absorbing_boundary() {};

  /**
   * @brief Constructor for absorbing boundaries
   *
   * This struct holds views with the names "ispec", "ijk", "jacobian2Dw",
   * and "normal" and the mass matrices are created if the medium is elastic or
   * acoustic under the names "mass_elastic" and "mass_acoustic" respectively.
   *
   * @param nglob Number of GLL points
   * @param num_abs_boundary_faces Number of boundary faces
   * @param ngllsquare Number of GLL points squared
   * @param acoustic Flag for acoustic simulation
   * @param elastic Flag for elastic simulation
   * @param nspec2D_xmin Number of elements on the x-min boundary
   * @param nspec2D_xmax Number of elements on the x-max boundary
   * @param nspec2D_ymin Number of elements on the y-min boundary
   * @param nspec2D_ymax Number of elements on the y-max boundary
   * @param NSPEC2D_BOTTOM Number of elements on the bottom boundary
   * @param NSPEC2D_TOP Number of elements on the top boundary
   *
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * const int nglob = 10;
   * const int num_abs_boundary_faces = 5;
   * const int ngllsquare = 100;
   * const bool acoustic = true;
   * const bool elastic = false;
   * const int nspec2D_xmin = 10;
   * const int nspec2D_xmax = 10;
   * const int nspec2D_ymin = 10;
   * const int nspec2D_ymax = 10;
   * const int NSPEC2D_BOTTOM = 10;
   * const int NSPEC2D_TOP = 10;
   *
   * absorbing_boundary<specfem::dimension::type::dim3> abs_boundary(nglob,
   *      num_abs_boundary_faces, ngllsquare, acoustic, elastic, nspec2D_xmin,
   *      nspec2D_xmax, nspec2D_ymin, nspec2D_ymax,
   *      NSPEC2D_BOTTOM, NSPEC2D_TOP);
   *
   * // Populate the views from the binary file
   * specfem::io::mesh::impl::fortran::dim3::read_index_array(stream,
   * abs_boundary.ispec);
   * specfem::io::mesh::impl::fortran::dim3::read_index_array(stream,
   * abs_boundary.ijk);
   * specfem::io::mesh::impl::fortran::dim3::read_array(stream,
   * abs_boundary.jacobian2Dw);
   * specfem::io::mesh::impl::fortran::dim3::read_array(stream,
   * abs_boundary.normal);
   * @endcode
   */
  absorbing_boundary(const int nglob, const int num_abs_boundary_faces,
                     const int ngllsquare, const bool acoustic,
                     const bool elastic, const int nspec2D_xmin,
                     const int nspec2D_xmax, const int nspec2D_ymin,
                     const int nspec2D_ymax, const int NSPEC2D_BOTTOM,
                     const int NSPEC2D_TOP)
      : ngllsquare(ngllsquare), num_abs_boundary_faces(num_abs_boundary_faces),
        acoustic(acoustic), elastic(elastic), nspec2D_xmin(nspec2D_xmin),
        nspec2D_xmax(nspec2D_xmax), nspec2D_ymin(nspec2D_ymin),
        nspec2D_ymax(nspec2D_ymax), NSPEC2D_BOTTOM(NSPEC2D_BOTTOM),
        NSPEC2D_TOP(NSPEC2D_TOP) {

    ispec =
        Kokkos::View<int *, Kokkos::HostSpace>("ispec", num_abs_boundary_faces);
    ijk = Kokkos::View<int ***, Kokkos::HostSpace>(
        "ijk", num_abs_boundary_faces, 3, ngllsquare);
    jacobian2Dw = Kokkos::View<type_real **, Kokkos::HostSpace>(
        "jacobian2Dw", num_abs_boundary_faces, ngllsquare);
    // ndim=3

    normal = Kokkos::View<type_real ***, Kokkos::HostSpace>(
        "normal", num_abs_boundary_faces, 3, ngllsquare);

    if (elastic) {
      mass_elastic = stacey_mass<specfem::element::medium_tag::elastic>(nglob);
    }
    if (acoustic) {
      mass_acoustic =
          stacey_mass<specfem::element::medium_tag::acoustic>(nglob);
    }

    // Create the boundary view elements only if the number of elements is
    // greater than 0 on said boundary
    if (nspec2D_xmin > 0) {
      ibelm_xmin =
          Kokkos::View<int *, Kokkos::HostSpace>("ibelm_xmin", nspec2D_xmin);
    }
    if (nspec2D_xmax > 0) {
      ibelm_xmax =
          Kokkos::View<int *, Kokkos::HostSpace>("ibelm_xmax", nspec2D_xmax);
    }
    if (nspec2D_ymin > 0) {
      ibelm_ymin =
          Kokkos::View<int *, Kokkos::HostSpace>("ibelm_ymin", nspec2D_ymin);
    }
    if (nspec2D_ymax > 0) {
      ibelm_ymax =
          Kokkos::View<int *, Kokkos::HostSpace>("ibelm_ymax", nspec2D_ymax);
    }
    if (NSPEC2D_BOTTOM > 0) {
      ibelm_bottom = Kokkos::View<int *, Kokkos::HostSpace>("ibelm_bottom",
                                                            NSPEC2D_BOTTOM);
    }
    if (NSPEC2D_TOP > 0) {
      ibelm_top =
          Kokkos::View<int *, Kokkos::HostSpace>("ibelm_top", NSPEC2D_TOP);
    }
  }
  /** @} */ // Constructors

  /**
   * @brief Print basic information on the absorbing boundary struct
   *
   */
  std::string print() const;

  /**
   * @brief Print the absorbing boundary struct
   *
   * @param iface index of the face.
   *
   */
  std::string print_ijk(const int iface) const;
};

} // namespace mesh
} // namespace specfem
