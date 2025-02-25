#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store element types for a 3D mesh
 *
 */
template <> struct element_types<specfem::dimension::type::dim3> {

  constexpr static auto dimension = specfem::dimension::type::dim3;

  int nspec; ///< Number of spectral elements

  int nelastic;     ///< Number of elastic spectral elements
  int nacoustic;    ///< Number of acoustic spectral elements
  int nporoelastic; ///< Number of poroelastic spectral elements

  template <typename T> using View1D = Kokkos::View<T *, Kokkos::HostSpace>;

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor initializing an empty struct
   *
   */
  element_types() = default;

  /**
   * @brief Construct a new element types object
   *
   * @param nspec Number of spectral elements
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * int nspec = 10;
   * specfem::mesh::element_types<specfem::dimension::type::dim3>
   *   element_types(nspec);
   * @endcode
   */
  element_types(const int nspec)
      : nspec(nspec), ispec_is_elastic("ispec_is_elastic", nspec),
        ispec_is_acoustic("ispec_is_acoustic", nspec),
        ispec_is_poroelastic("ispec_is_poroelastic", nspec){};

  ///@}

  /**
   * @brief Set the elements for the given medium type
   *
   * This function loops over the boolean arrays ispec_is_elastic,
   * ispec_is_acoustic, and ispec_is_poroelastic and creates arrays that contain
   * the indices of the spectral elements that are elastic, acoustic, and
   * poroelastic.
   *
   * @see get_elements
   */
  void set_elements();

  /**
   * @brief Get the elements for the given medium type
   *
   * @tparam MediumTag Medium type
   * @return View1D<int> Element indices for the given medium type
   */
  template <specfem::element::medium_tag MediumTag> View1D<int> get_elements();

  /**
   * @brief Print basic information about the element types
   *
   */
  void print() const;

  /**
   * @brief Print the element type of a specific spectral element
   *
   * @param ispec Index of the spectral element
   */
  void print(const int ispec) const;

  /**
   * @brief Print the element type of element i/n<medium>
   *
   * @tparam MediumTag Medium type
   */
  template <specfem::element::medium_tag MediumTag>
  void print(const int i) const;

  View1D<bool> ispec_is_elastic;     ///< Elastic spectral elements
  View1D<bool> ispec_is_acoustic;    ///< Acoustic spectral elements
  View1D<bool> ispec_is_poroelastic; ///< Poroelastic spectral elements

private:
  View1D<int> ispec_elastic;     ///< Elastic spectral elements
  View1D<int> ispec_acoustic;    ///< Acoustic spectral elements
  View1D<int> ispec_poroelastic; ///< Poroelastic spectral elements
};
} // namespace mesh
} // namespace specfem
