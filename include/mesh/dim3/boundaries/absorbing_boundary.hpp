#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

template <specfem::element::medium_tag MediumTag> struct stacey_mass {};

template <> struct stacey_mass<specfem::element::medium_tag::elastic> {
  int nglob;             ///< Number of GLL points
  bool acoustic = false; ///< Flag for acoustic simulation
  bool elastic = true;   ///< Flag for elastic simulation

  Kokkos::View<type_real *, Kokkos::HostSpace> x;
  Kokkos::View<type_real *, Kokkos::HostSpace> y;
  Kokkos::View<type_real *, Kokkos::HostSpace> z;

  // Default constructor
  stacey_mass(){};

  stacey_mass(const int nglob) : nglob(nglob) {
    x = Kokkos::View<type_real *, Kokkos::HostSpace>("rmass_x", nglob);
    y = Kokkos::View<type_real *, Kokkos::HostSpace>("rmass_y", nglob);
    z = Kokkos::View<type_real *, Kokkos::HostSpace>("rmass_z", nglob);
  }
};

template <> struct stacey_mass<specfem::element::medium_tag::acoustic> {
  int nglob;            ///< Number of GLL points
  bool acoustic = true; ///< Flag for acoustic simulation
  bool elastic = false; ///< Flag for elastic simulation

  Kokkos::View<type_real *, Kokkos::HostSpace> mass;

  // Default constructor
  stacey_mass(){};

  stacey_mass(const int nglob) : nglob(nglob) {
    mass = Kokkos::View<type_real *, Kokkos::HostSpace>("rmass_x", nglob);
  }
};

template <> struct absorbing_boundary<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type
  // const int ndim = 3; ///< Dimension

  int num_abs_boundary_faces; ///< Number of boundaries faces
  int nelements;              ///< Number of elements on the boundary
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

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  absorbing_boundary(){};

  absorbing_boundary(const int nglob, const int num_abs_boundary_faces,
                     const int ngllsquare, const bool acoustic,
                     const bool elastic, const int nspec2D_xmin,
                     const int nspec2D_xmax, const int nspec2D_ymin,
                     const int nspec2D_ymax, const int NSPEC2D_BOTTOM,
                     const int NSPEC2D_TOP)
      : nelements(num_abs_boundary_faces), ngllsquare(ngllsquare),
        num_abs_boundary_faces(num_abs_boundary_faces), acoustic(acoustic),
        elastic(elastic), nspec2D_xmin(nspec2D_xmin),
        nspec2D_xmax(nspec2D_xmax), nspec2D_ymin(nspec2D_ymin),
        nspec2D_ymax(nspec2D_ymax), NSPEC2D_BOTTOM(NSPEC2D_BOTTOM),
        NSPEC2D_TOP(NSPEC2D_TOP) {

    ispec = Kokkos::View<int *, Kokkos::HostSpace>("ispec", nelements);
    ijk = Kokkos::View<int ***, Kokkos::HostSpace>("ijk", nelements, 3,
                                                   ngllsquare);
    jacobian2Dw = Kokkos::View<type_real **, Kokkos::HostSpace>(
        "jacobian2Dw", nelements, ngllsquare);
    // ndim=3

    normal = Kokkos::View<type_real ***, Kokkos::HostSpace>("normal", nelements,
                                                            3, ngllsquare);

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
  ///@}

  void print() const;
};

} // namespace mesh
} // namespace specfem
