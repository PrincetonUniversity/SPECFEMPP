#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {
/**
 * @brief Inner outer element medium struct for dim3
 *
 */
template <specfem::element::medium_tag MediumTag> struct inner_outer_medium {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

  int nspec_inner;     ///< Number of inner elements
  int nspec_outer;     ///< Number of outer elements
  int num_phase_ispec; ///< Number of phase elements

  Kokkos::View<int **, Kokkos::HostSpace> phase_ispec_inner; ///< Inner elements

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  inner_outer_medium() {};

  /**
   * @brief Constructor
   *
   * Constructor for the inner_outer struct that initializes the number of
   * the underlying arrays with the given parameters
   *
   * @param nspec_inner Number of inner elements
   * @param nspec_outer Number of outer elements
   * @param num_phase_ispec Number of phase elements
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * specfem::mesh::inner_outer<specfem::dimension::type::dim3>
   * inner_outer(nspec_inner, nspec_outer, num_phase_ispec);
   * @endcode
   *
   */
  inner_outer_medium(const int nspec_inner, const int nspec_outer,
                     const int num_phase_ispec)
      : nspec_inner(nspec_inner), nspec_outer(nspec_outer),
        num_phase_ispec(num_phase_ispec),
        phase_ispec_inner("phase_ispec_inner", num_phase_ispec, 2) {};

  ///@}
};

/**
 * @brief Inner outer element struct for dim3
 *
 */
template <> struct inner_outer<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

  int nspec;        ///< Number of spectral elements
  bool acoustic;    ///< If simulation/mesh is acoustic
  bool elastic;     ///< If simulation/mesh is elastic
  bool poroelastic; ///< If simulation/mesh is poroelastic

  Kokkos::View<bool *, Kokkos::HostSpace> ispec_is_inner; ///< Inner elements

  inner_outer_medium<specfem::element::medium_tag::elastic>
      inner_outer_elastic; ///< Inner outer elements
  inner_outer_medium<specfem::element::medium_tag::acoustic>
      inner_outer_acoustic; ///< Inner outer elements
  inner_outer_medium<specfem::element::medium_tag::poroelastic>
      inner_outer_poroelastic; ///< Inner outer elements

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor initializing an empty struct
   *
   */
  inner_outer() {};

  /**
   * @brief Construct a new inner outer object
   *
   * @param nspec Number of spectral elements
   * @param acoustic If simulation/mesh is acoustic
   * @param elastic If simulation/mesh is elastic
   * @param poroelastic If simulation/mesh is poroelastic
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * specfem::mesh::inner_outer<specfem::dimension::type::dim3>
   *    inner_outer(nspec_inner, nspec_outer, num_phase_ispec);
   * @endcode
   */
  inner_outer(const int nspec, const int acoustic, const int elastic,
              const int poroelastic)
      : nspec(nspec), acoustic(acoustic), elastic(elastic),
        poroelastic(poroelastic), ispec_is_inner("ispec_is_inner", nspec) {};
};

} // namespace mesh
} // namespace specfem
