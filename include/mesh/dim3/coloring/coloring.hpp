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
template <specfem::element::medium_tag MediumTag> struct medium_coloring {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

  int num_colors_outer,
      num_colors_inner; ///< Number of colors for inner and outer elements

  Kokkos::View<int *, Kokkos::HostSpace> elements; ///< Colors for
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  medium_coloring() {};

  /**
   * @brief Constructor for the medium_coloring struct that initializes the
   * number of the underlying arrays with the given parameters
   *
   * @param num_colors_outer Number of colors for outer elements
   * @param num_colors_inner Number of colors for inner elements
   *
   */
  medium_coloring(const int num_colors_outer, const int num_colors_inner)
      : num_colors_outer(num_colors_outer), num_colors_inner(num_colors_inner),
        elements("elements", num_colors_outer + num_colors_inner) {};

  ///@}
};

/**
 * @brief Inner outer element struct for dim3
 *
 */
template <> struct coloring<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

  bool acoustic_simulation; ///< Acoustic simulation
  bool elastic_simulation;  ///< Elastic simulation

  medium_coloring<specfem::element::medium_tag::acoustic>
      coloring_acoustic; ///< acousting coloring
  medium_coloring<specfem::element::medium_tag::elastic>
      coloring_elastic; ///< coloring elastic

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor initializing an empty struct
   *
   */
  coloring() {};

  /**
   * @brief Construct a new coloring object from the given parameters
   *
   */
  coloring(const bool acoustic_simulation, const bool elastic_simulation)
      : acoustic_simulation(acoustic_simulation),
        elastic_simulation(elastic_simulation) {};

  ///@}
};

} // namespace mesh
} // namespace specfem
