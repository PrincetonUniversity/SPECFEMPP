#pragma once

#include "enumerations/dimension.hpp"
#include "specfem/mesh/mesh_base.hpp"

namespace specfem {

namespace mesh {
/**
 * @brief Mesh parameters
 *
 * @tparam DimensionTag Dimension type for the mesh (2D or 3D)
 */
template <specfem::dimension::type DimensionTag> struct parameters;

/**
 * @brief Template specialization for 2D mesh parameters
 */
template <> struct parameters<specfem::dimension::type::dim2> {
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension
                                      ///< type
  int numat;                          ///< Total number of different materials
  int ngnod;                          ///< Number of control nodes
  int nspec;                          ///< Number of spectral elements
  int pointsdisp;      // Total number of points to display (Only used for
                       // visualization)
  int nelemabs;        ///< Number of elements on absorbing boundary
  int nelem_acforcing; ///< Number of elements on acoustic forcing boundary
  int nelem_acoustic_surface;  ///< Number of elements on acoustic surface
  int num_fluid_solid_edges;   ///< Number of solid-fluid edges
  int num_fluid_poro_edges;    ///< Number of fluid-poroelastic edges
  int num_solid_poro_edges;    ///< Number of solid-poroelastic edges
  int nnodes_tangential_curve; ///< Number of elements on tangential curve
  int nelem_on_the_axis;       ///< Number of axial elements
  bool plot_lowerleft_corner_only;

  /**
   * @brief Default constructor
   *
   */
  parameters() {};

  /**
   * @brief Construct a properties object
   *
   * @param numat Total number of materials
   * @param ngnod Total number of control nodes
   * @param nspec Total number of spectral elements
   * @param pointsdisp Total number of points to display
   * @param nelemabs Number of elements on absorbing boundary
   * @param nelem_acforcing Number of elements on acoustic forcing boundary
   * @param nelem_acoustic_surface Number of elements on acoustic surface
   * @param num_fluid_solid_edges Number of solid-fluid edges
   * @param num_fluid_poro_edges Number of fluid-poroelastic edges
   * @param num_solid_poro_edges Number of solid-poroelastic edges
   * @param nnodes_tangential_curve Number of elements on tangential curve
   * @param nelem_on_the_axis Number of axial elements
   * @param plot_lowerleft_corner_only Flag to plot only lower left corner
   */

  parameters(const int numat, const int ngnod, const int nspec,
             const int pointsdisp, const int nelemabs,
             const int nelem_acforcing, const int nelem_acoustic_surface,
             const int num_fluid_solid_edges, const int num_fluid_poro_edges,
             const int num_solid_poro_edges, const int nnodes_tangential_curve,
             const int nelem_on_the_axis, const bool plot_lowerleft_corner_only)
      : numat(numat), ngnod(ngnod), nspec(nspec), pointsdisp(pointsdisp),
        nelemabs(nelemabs), nelem_acforcing(nelem_acforcing),
        nelem_acoustic_surface(nelem_acoustic_surface),
        num_fluid_solid_edges(num_fluid_solid_edges),
        num_fluid_poro_edges(num_fluid_poro_edges),
        num_solid_poro_edges(num_solid_poro_edges),
        nnodes_tangential_curve(nnodes_tangential_curve),
        nelem_on_the_axis(nelem_on_the_axis),
        plot_lowerleft_corner_only(plot_lowerleft_corner_only) {};
};

} // namespace mesh
} // namespace specfem
