#pragma once

#include "mesh/mesh_base.hpp"

namespace specfem {
namespace mesh {

/**
 * @brief Template specialization for 3D mesh parameters
 */
template <> struct parameters<specfem::dimension::type::dim3> {
  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension
                                      ///< type

  // Flags
  bool acoustic_simulation;    ///< Flag for acoustic simulation
  bool elastic_simulation;     ///< Flag for elastic simulation
  bool poroelastic_simulation; ///< Flag for poroelastic simulation
  bool anisotropy;             ///< Flag for anisotropy
  bool stacey_abc;             ///< Flag for Stacey absorbing boundary c.
  bool pml_abc;                ///< Flag for PML absorbing boundary c.
  bool approximate_ocean_load; ///< Flag for approx. ocean load
  bool use_mesh_coloring;      ///< Flag for mesh coloring

  // Integer Parameters: Dimensions and GLL layouts
  int ndim;       ///< Number of dimensions
  int ngllx;      ///< Number of GLL points in x
  int nglly;      ///< Number of GLL points in y
  int ngllz;      ///< Number of GLL points in z
  int ngllsquare; ///< Number of GLL points in square
  int nproc;      ///< Number of processors
  int ngnod;      ///< Number of control nodes per spectral element
  int nnodes;     ///< Number of control nodes

  // Integer Parameters: Elements/Nodes
  int nspec;           ///< Number of spectral elements (SEs)
  int nspec_poro;      ///< Number of poroelastic SEs
  int nglob;           ///< Number of global nodes
  int nglob_ocean;     ///< Number of global ocean nodes
  int nspec2D_bottom;  ///< Number of 2D SEs at the bottom
  int nspec2D_top;     ///< Number of 2D SEs at the top
  int nspec2D_xmin;    ///< Number of 2D SEs at the left
  int nspec2D_xmax;    ///< Number of 2D SEs at the right
  int nspec2D_ymin;    ///< Number of 2D SEs at the front
  int nspec2D_ymax;    ///< Number of 2D SEs at the back
  int nspec_irregular; ///< Number of irregular SEs

  // Integer Parameters: Mesh
  int num_neighbors;                  ///< Number of neighbors
  int nfaces_surface;                 ///< Number of faces on the surface
  int num_abs_boundary_faces;         ///< Number of absorbing boundary faces
  int num_free_surface_faces;         ///< Number of free surface faces
  int num_coupling_ac_el_faces;       ///< Number of acoustic-elastic faces
  int num_coupling_ac_po_faces;       ///< Number of acoustic-poroelastic faces
  int num_coupling_el_po_faces;       ///< Number of elastic-poroelastic faces
  int num_coupling_po_el_faces;       ///< Number of poroelastic-elastic faces
  int num_interfaces_ext_mesh;        ///< Number of external mesh interfaces
  int max_nibool_interfaces_ext_mesh; ///< Maximum number of interfaces

  // Integer Parameters: Nspec Inner/Outer
  int nspec_inner_acoustic;    ///< Number of inner acoustic SEs
  int nspec_outer_acoustic;    ///< Number of outer acoustic SEs
  int nspec_inner_elastic;     ///< Number of inner elastic SEs
  int nspec_outer_elastic;     ///< Number of outer elastic SEs
  int nspec_inner_poroelastic; ///< Number of inner poroelastic SEs
  int nspec_outer_poroelastic; ///< Number of outer poroelastic SEs

  // Integer Parameters: coloring
  int num_phase_ispec_acoustic;    ///< Number of phase ispec acoustic
  int num_phase_ispec_elastic;     ///< Number of phase ispec elastic
  int num_phase_ispec_poroelastic; ///< Number of phase ispec poroelastic
  int num_colors_inner_acoustic;   ///< Number of colors inner acoustic
  int num_colors_outer_acoustic;   ///< Number of colors outer acoustic
  int num_colors_inner_elastic;    ///< Number of colors inner elastic
  int num_colors_outer_elastic;    ///< Number of colors outer elastic

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  parameters() {};

  /**
   *
   * @brief Construct a new parameters object
   *
   * @param acoustic_simulation Flag for acoustic simulation
   * @param elastic_simulation Flag for elastic simulation
   * @param poroelastic_simulation Flag for poroelastic simulation
   * @param anisotropy Flag for anisotropy
   * @param stacey_abc Flag for Stacey absorbing boundary c.
   * @param pml_abc Flag for PML absorbing boundary c.
   * @param approximate_ocean_load Flag for approx. ocean load
   * @param use_mesh_coloring Flag for mesh coloring
   *
   *
   *
   * @param nspec Number of spectral elements (SEs)
   * @param nspec_poro Number of poroelastic SEs
   * @param nglob Number of global nodes
   * @param nglob_ocean Number of global ocean nodes
   * @param nspec2D_bottom Number of 2D SEs at the bottom
   * @param nspec2D_top Number of 2D SEs at the top
   * @param nspec2D_xmin Number of 2D SEs at the left
   * @param nspec2D_xmax Number of 2D SEs at the right
   * @param nspec2D_ymin Number of 2D SEs at the front
   * @param nspec2D_ymax Number of 2D SEs at the back
   * @param nspec_irregular Number of irregular SEs
   *
   * @param num_neighbors Number of neighbors
   * @param nfaces_surface Number of faces on the surface
   * @param num_abs_boundary_faces Number of absorbing boundary faces
   * @param num_free_surface_faces Number of free surface faces
   * @param num_coupling_ac_el_faces Number of acoustic-elastic faces
   * @param num_coupling_ac_po_faces Number of acoustic-poroelastic faces
   * @param num_coupling_el_po_faces Number of elastic-poroelastic faces
   * @param num_coupling_po_el_faces Number of poroelastic-elastic faces
   * @param num_interfaces_ext_mesh Number of external mesh interfaces
   * @param max_nibool_interfaces_ext_mesh Maximum number of interfaces
   *
   * @param nspec_inner_acoustic Number of inner acoustic SEs
   * @param nspec_outer_acoustic Number of outer acoustic SEs
   * @param nspec_inner_elastic Number of inner elastic SEs
   * @param nspec_outer_elastic Number of outer elastic SEs
   * @param nspec_inner_poroelastic Number of inner poroelastic SEs
   * @param nspec_outer_poroelastic Number of outer poroelastic SEs
   *
   * @param num_phase_ispec_acoustic Number of phase ispec acoustic
   * @param num_phase_ispec_elastic Number of phase ispec elastic
   * @param num_phase_ispec_poroelastic Number of phase ispec poroelastic
   * @param num_colors_inner_acoustic Number of colors inner acoustic
   * @param num_colors_outer_acoustic Number of colors outer acoustic
   * @param num_colors_inner_elastic Number of colors inner elastic
   * @param num_colors_outer_elastic Number of colors outer elastic
   *
   */
  parameters(
      const bool acoustic_simulation, const bool elastic_simulation,
      const bool poroelastic_simulation, const bool anisotropy,
      const bool stacey_abc, const bool pml_abc,
      const bool approximate_ocean_load, const bool use_mesh_coloring,
      const int ndim, const int ngllx, const int nglly, const int ngllz,
      const int ngllsquare, const int nproc, const int nspec,
      const int nspec_poro, const int nglob, const int nglob_ocean,
      const int nspec2D_bottom, const int nspec2D_top, const int nspec2D_xmin,
      const int nspec2D_xmax, const int nspec2D_ymin, const int nspec2D_ymax,
      const int nspec_irregular, const int num_neighbors,
      const int nfaces_surface, const int num_abs_boundary_faces,
      const int num_free_surface_faces, const int num_coupling_ac_el_faces,
      const int num_coupling_ac_po_faces, const int num_coupling_el_po_faces,
      const int num_coupling_po_el_faces, const int num_interfaces_ext_mesh,
      const int max_nibool_interfaces_ext_mesh, const int nspec_inner_acoustic,
      const int nspec_outer_acoustic, const int nspec_inner_elastic,
      const int nspec_outer_elastic, const int nspec_inner_poroelastic,
      const int nspec_outer_poroelastic, const int num_phase_ispec_acoustic,
      const int num_phase_ispec_elastic, const int num_phase_ispec_poroelastic,
      const int num_colors_inner_acoustic, const int num_colors_outer_acoustic,
      const int num_colors_inner_elastic, const int num_colors_outer_elastic)
      : acoustic_simulation(acoustic_simulation),
        elastic_simulation(elastic_simulation),
        poroelastic_simulation(poroelastic_simulation), anisotropy(anisotropy),
        stacey_abc(stacey_abc), pml_abc(pml_abc),
        approximate_ocean_load(approximate_ocean_load),
        use_mesh_coloring(use_mesh_coloring), ndim(ndim), ngllx(ngllx),
        nglly(nglly), ngllz(ngllz), ngllsquare(ngllsquare), nproc(nproc),
        nspec(nspec), nspec_poro(nspec_poro), nglob(nglob),
        nglob_ocean(nglob_ocean), nspec2D_bottom(nspec2D_bottom),
        nspec2D_top(nspec2D_top), nspec2D_xmin(nspec2D_xmin),
        nspec2D_xmax(nspec2D_xmax), nspec2D_ymin(nspec2D_ymin),
        nspec2D_ymax(nspec2D_ymax), nspec_irregular(nspec_irregular),
        num_neighbors(num_neighbors), nfaces_surface(nfaces_surface),
        num_abs_boundary_faces(num_abs_boundary_faces),
        num_free_surface_faces(num_free_surface_faces),
        num_coupling_ac_el_faces(num_coupling_ac_el_faces),
        num_coupling_ac_po_faces(num_coupling_ac_po_faces),
        num_coupling_el_po_faces(num_coupling_el_po_faces),
        num_coupling_po_el_faces(num_coupling_po_el_faces),
        num_interfaces_ext_mesh(num_interfaces_ext_mesh),
        max_nibool_interfaces_ext_mesh(max_nibool_interfaces_ext_mesh),
        nspec_inner_acoustic(nspec_inner_acoustic),
        nspec_outer_acoustic(nspec_outer_acoustic),
        nspec_inner_elastic(nspec_inner_elastic),
        nspec_outer_elastic(nspec_outer_elastic),
        nspec_inner_poroelastic(nspec_inner_poroelastic),
        nspec_outer_poroelastic(nspec_outer_poroelastic),
        num_phase_ispec_acoustic(num_phase_ispec_acoustic),
        num_phase_ispec_elastic(num_phase_ispec_elastic),
        num_phase_ispec_poroelastic(num_phase_ispec_poroelastic),
        num_colors_inner_acoustic(num_colors_inner_acoustic),
        num_colors_outer_acoustic(num_colors_outer_acoustic),
        num_colors_inner_elastic(num_colors_inner_elastic),
        num_colors_outer_elastic(num_colors_outer_elastic) {};

  ///@}

  /**
   * @brief Print basic information about the parameters
   *
   */
  std::string print() const;
};

} // namespace mesh
} // namespace specfem
