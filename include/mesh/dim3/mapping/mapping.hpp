#pragma once

#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store mapping for a 3D mesh
 *
 */
template <> struct mapping<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension

  using UniqueViewInt = Kokkos::View<int *, Kokkos::HostSpace>;
  using UniqueViewBool = Kokkos::View<bool *, Kokkos::HostSpace>;
  using LocalViewInt =
      Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  // Parameters needed for ibool mapping
  int nspec;           ///< Number of spectral elements
  int nglob;           ///< Number of global elements
  int nspec_irregular; ///< Number of irregular elements

  int ngllx; ///< Number of GLL points in x direction
  int nglly; ///< Number of GLL points in y direction
  int ngllz; ///< Number of GLL points in z direction

  // I do not know currently what these are used for
  type_real xix_regular;      ///< Regular xi value
  type_real jacobian_regular; ///< Regular Jacobian value

  // Indices of irregular elements size nspec_irregular
  UniqueViewInt irregular_elements; ///< Irregular elements

  // ibool size nspec, ngllx, nglly, ngllz
  LocalViewInt ibool; ///< The local to global mapping

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new mapping object
   *
   */
  mapping() {}; // Default constructor

  /**
   * @brief Construct a new mapping object
   *
   * @param nspec Number of spectral elements
   * @param nglob Number of global nodes
   * @param nspec_irregular Number of irregular elements
   * @param ngllx Number of GLL points in x direction
   * @param nglly Number of GLL points in y direction
   * @param ngllz Number of GLL points in z direction
   */
  mapping(int nspec, int nglob, int nspec_irregular, int ngllz, int nglly,
          int ngllx)
      : nspec(nspec), nglob(nglob), nspec_irregular(nspec_irregular),
        ngllx(ngllx), nglly(nglly), ngllz(ngllz), xix_regular(0.0),
        jacobian_regular(0.0),
        irregular_elements("irregular_elements", nspec_irregular),
        ibool("ibool", nspec, ngllz, nglly, ngllx) {};
  ///@}

  /**
   * @brief Print basic information about the mapping
   *
   */
  std::string print() const;

  /**
   * @brief Print the mapping for the given spectral element
   *
   * @param ispec Spectral element index
   */
  std::string print(const int ispec) const;
};

} // namespace mesh
} // namespace specfem
