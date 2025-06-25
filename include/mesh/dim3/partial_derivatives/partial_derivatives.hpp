#pragma once
#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store partial derivatives for a 3D mesh
 *
 */
template <> struct partial_derivatives<specfem::dimension::type::dim3> {
  constexpr static auto dimension = specfem::dimension::type::dim3;

  using LocalView = Kokkos::View<type_real ****, Kokkos::HostSpace>;

  // Parameters
  int nspec; ///< Number of spectral elements
  int ngllx; ///< Number of GLL points in x
  int nglly; ///< Number of GLL points in y
  int ngllz; ///< Number of GLL points in z

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
  partial_derivatives() {}; // Default constructor

  /**
   * @brief Construct a new partial derivatives object
   *
   * @param nspec Number of spectral elements
   * @param ngllx Number of GLL points in x
   * @param nglly Number of GLL points in y
   * @param ngllz Number of GLL points in z
   *
   *
   */
  partial_derivatives(int nspec, int ngllx, int nglly, int ngllz)
      : nspec(nspec), ngllx(ngllx), nglly(nglly), ngllz(ngllz),
        xix("xix", nspec, ngllx, nglly, ngllz),
        xiy("xiy", nspec, ngllx, nglly, ngllz),
        xiz("xiz", nspec, ngllx, nglly, ngllz),
        etax("etax", nspec, ngllx, nglly, ngllz),
        etay("etay", nspec, ngllx, nglly, ngllz),
        etaz("etaz", nspec, ngllx, nglly, ngllz),
        gammax("gammax", nspec, ngllx, nglly, ngllz),
        gammay("gammay", nspec, ngllx, nglly, ngllz),
        gammaz("gammaz", nspec, ngllx, nglly, ngllz),
        jacobian("jacobian", nspec, ngllx, nglly, ngllz) {};

  ///@}

  /**
   * @brief Print basic information on the partial derivatives
   *
   */
  std::string print() const;

  /**
   * @brief Print the partial derivatives for a given spectral element and
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
   * @brief Print the partial derivatives for a given spectral element
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
