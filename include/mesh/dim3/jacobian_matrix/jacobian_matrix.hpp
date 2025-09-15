#pragma once
#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store Jacobian matrix for a 3D mesh
 *
 */
template <> struct jacobian_matrix<specfem::dimension::type::dim3> {
  constexpr static auto dimension = specfem::dimension::type::dim3;

  using LocalView = Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                 Kokkos::DefaultHostExecutionSpace>;

  using View1DInt = Kokkos::View<int *, Kokkos::LayoutLeft,
                                 Kokkos::DefaultHostExecutionSpace>;
  // Parameters
  int nspec;           ///< Number of spectral elements
  int ngllx;           ///< Number of GLL points in x
  int nglly;           ///< Number of GLL points in y
  int ngllz;           ///< Number of GLL points in z
  int nspec_irregular; ///< Number of irregular spectral elements

  // Values
  type_real xix_regular;
  type_real jacobian_regular;

  View1DInt irregular_element_number; ///< 1D Kokkos::view of type int for
                                      ///< irregular_element_number

  LocalView xix;      ///< 4D Kokkos::view of type real for xix
  LocalView xiy;      ///< 4D Kokkos::view of type real for xiy
  LocalView xiz;      ///< 4D Kokkos::view of type real for xiz
  LocalView etax;     ///< 4D Kokkos::view of type real for etax
  LocalView etay;     ///< 4D Kokkos::view of type real for etay
  LocalView etaz;     ///< 4D Kokkos::view of type real for etaz
  LocalView gammax;   ///< 4D Kokkos::view of type real for gammax
  LocalView gammay;   ///< 4D Kokkos::view of type real for gammay
  LocalView gammaz;   ///< 4D Kokkos::view of type real for gammaz
  LocalView jacobian; ///< 4D Kokkos::view of type real for jacobian

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  jacobian_matrix() {}; // Default constructor

  /**
   * @brief Construct a new Jacobian matrix object
   *
   * @param nspec Number of spectral elements
   * @param ngllx Number of GLL points in x
   * @param nglly Number of GLL points in y
   * @param ngllz Number of GLL points in z
   *
   *
   */
  jacobian_matrix(int nspec, int ngllz, int nglly, int ngllx)
      : nspec(nspec), ngllx(ngllx), nglly(nglly), ngllz(ngllz),
        xix("xix", nspec, ngllz, nglly, ngllx),
        xiy("xiy", nspec, ngllz, nglly, ngllx),
        xiz("xiz", nspec, ngllz, nglly, ngllx),
        etax("etax", nspec, ngllz, nglly, ngllx),
        etay("etay", nspec, ngllz, nglly, ngllx),
        etaz("etaz", nspec, ngllz, nglly, ngllx),
        gammax("gammax", nspec, ngllz, nglly, ngllx),
        gammay("gammay", nspec, ngllz, nglly, ngllx),
        gammaz("gammaz", nspec, ngllz, nglly, ngllx),
        jacobian("jacobian", nspec, ngllz, nglly, ngllx) {};

  ///@}

  /**
   * @brief Print basic information on the Jacobian matrix
   *
   */
  std::string print() const;

  /**
   * @brief Print the Jacobian matrix for a given spectral element and
   *        GLL point
   *
   * @param ispec Spectral element index
   * @param igllx GLL point index in x
   * @param iglly GLL point index in y
   * @param igllz GLL point index in z
   *
   * @return std::string
   */
  std::string print(int ispec, int igllx, int iglly, int igllz) const;

  /**
   * @brief Print the Jacobian matrix for a given spectral element
   *
   * @param ispec Spectral element index
   * @param component Component to print (xix, xiy, xiz, etax, etay, etaz,
   *                  gammax, gammay, gammaz, jacobian)
   *
   * @return std::string
   */
  std::string print(int ispec, const std::string) const;
};

} // namespace mesh
} // namespace specfem
