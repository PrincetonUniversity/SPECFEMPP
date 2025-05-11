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

  int nacoustic;    ///< Number of acoustic spectral elements
  int nelastic;     ///< Number of elastic spectral elements
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
   * @param nacoustic Number of acoustic spectral elements
   * @param nelastic Number of elastic spectral elements
   * @param nporoelastic Number of poroelastic spectral elements
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * int nspec = 10;
   * specfem::mesh::element_types<specfem::dimension::type::dim3>
   *   element_types(nspec);
   * @endcode
   */
  element_types(const int nspec, const int nacoustic, const int nelastic,
                const int nporoelastic)
      : nspec(nspec), nacoustic(nacoustic), nelastic(nelastic),
        nporoelastic(nporoelastic), ispec_type("ispec_type", nspec),
        ispec_acoustic("mesh.element_types.ispec_acoustic", nacoustic),
        ispec_elastic("mesh.element_types.ispec_elastic", nelastic),
        ispec_poroelastic("mesh.element_types.ispec_poroelastic",
                          nporoelastic) {};

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
  void set_elements(View1D<int> &ispec_type_in);

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
  std::string print() const;

  /**
   * @brief Print the element type of a specific spectral element
   *
   * @param ispec Index of the spectral element
   */
  std::string print(const int ispec) const;

  /**
   * @brief Print the element type of element i/n<medium>
   *
   * @tparam MediumTag Medium type
   */
  template <specfem::element::medium_tag MediumTag>
  std::string print(const int i) const;

  View1D<specfem::element::medium_tag>
      ispec_type; ///< Elastic spectral elements with:
                  ///< ispec_type[ispec] = 0 (acoustic)
                  ///< ispec_type[ispec] = 1 (elastic)
                  ///< ispec_type[ispec] = 2 (poroelastic)

private:
  View1D<int> ispec_elastic;     ///< Elastic to global mapping
  View1D<int> ispec_acoustic;    ///< Acoustic to global mapping
  View1D<int> ispec_poroelastic; ///< Poroelastic to global mapping
};
} // namespace mesh
} // namespace specfem
