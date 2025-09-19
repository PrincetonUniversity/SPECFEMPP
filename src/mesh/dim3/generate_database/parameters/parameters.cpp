#include "mesh/dim3/generate_database/parameters/parameters.hpp"
#include <iostream>
#include <sstream>

std::string
specfem::mesh::parameters<specfem::dimension::type::dim3>::print() const {

  std::ostringstream message;

  message << "Acoustic simulation:........................... "
          << acoustic_simulation << "\n";
  message << "Elastic simulation:............................ "
          << elastic_simulation << "\n";
  message << "Poroelastic simulation:........................ "
          << poroelastic_simulation << "\n";
  message << "Anisotropy:.................................... " << anisotropy
          << "\n";
  message << "Stacey ABC:.................................... " << stacey_abc
          << "\n";
  message << "PML ABC:....................................... " << pml_abc
          << "\n";
  message << "Approximate ocean load:........................ "
          << approximate_ocean_load << "\n";
  message << "Use mesh coloring:............................. "
          << use_mesh_coloring << "\n";
  message << "Number of dimensions:.......................... " << ndim << "\n";
  message << "Number of GLLX:................................ " << ngllx
          << "\n";
  message << "Number of GLLY:................................ " << nglly
          << "\n";
  message << "Number of GLLZ:................................ " << ngllz
          << "\n";
  message << "Number of GLL square:.......................... " << ngllsquare
          << "\n";
  message << "Number of processors:.......................... " << nproc
          << "\n";
  message << "Number of spectral elements:................... " << nspec
          << "\n";
  message << "Number of spectral elements for poroelastic:... " << nspec_poro
          << "\n";
  message << "Number of global nodes:........................ " << nglob
          << "\n";
  message << "Number of global nodes for ocean:.............. " << nglob_ocean
          << "\n";
  message << "Number of spectral elements for 2D bottom:..... "
          << nspec2D_bottom << "\n";
  message << "Number of spectral elements for 2D top:........ " << nspec2D_top
          << "\n";
  message << "Number of spectral elements for 2D xmin:....... " << nspec2D_xmin
          << "\n";
  message << "Number of spectral elements for 2D xmax:....... " << nspec2D_xmax
          << "\n";
  message << "Number of spectral elements for 2D ymin:....... " << nspec2D_ymin
          << "\n";
  message << "Number of spectral elements for 2D ymax:....... " << nspec2D_ymax
          << "\n";
  message << "Number of irregular spectral elements:......... "
          << nspec_irregular << "\n";
  message << "Number of neighbors:........................... " << num_neighbors
          << "\n";
  message << "Number of faces on the surface:................ "
          << nfaces_surface << "\n";
  message << "Number of absorbing boundary faces:............ "
          << num_abs_boundary_faces << "\n";
  message << "Number of free surface faces:.................. "
          << num_free_surface_faces << "\n";
  message << "Number of acoustic-elastic faces:.............. "
          << num_coupling_ac_el_faces << "\n";
  message << "Number of acoustic-poroelastic faces:.......... "
          << num_coupling_ac_po_faces << "\n";
  message << "Number of elastic-poroelastic faces:........... "
          << num_coupling_el_po_faces << "\n";
  message << "Number of poroelastic-elastic faces:........... "
          << num_coupling_po_el_faces << "\n";
  message << "Number of external mesh interfaces:............ "
          << num_interfaces_ext_mesh << "\n";
  message << "Maximum number of interfaces:.................. "
          << max_nibool_interfaces_ext_mesh << "\n";
  message << "Number of inner acoustic spectral elements:.... "
          << nspec_inner_acoustic << "\n";
  message << "Number of outer acoustic spectral elements:.... "
          << nspec_outer_acoustic << "\n";
  message << "Number of inner elastic spectral elements:..... "
          << nspec_inner_elastic << "\n";
  message << "Number of outer elastic spectral elements:..... "
          << nspec_outer_elastic << "\n";
  message << "Number of inner poroelastic spectral elements:. "
          << nspec_inner_poroelastic << "\n";
  message << "Number of outer poroelastic spectral elements:. "
          << nspec_outer_poroelastic << "\n";
  message << "Number of phase ispec acoustic: ............... "
          << num_phase_ispec_acoustic << "\n";
  message << "Number of phase ispec elastic:................. "
          << num_phase_ispec_elastic << "\n";
  message << "Number of phase ispec poroelastic:............. "
          << num_phase_ispec_poroelastic << "\n";
  message << "Number of colors inner acoustic:............... "
          << num_colors_inner_acoustic << "\n";
  message << "Number of colors outer acoustic:............... "
          << num_colors_outer_acoustic << "\n";
  message << "Number of colors inner elastic:................ "
          << num_colors_inner_elastic << "\n";
  message << "Number of colors outer elastic:................ "
          << num_colors_outer_elastic << "\n";

  return message.str();
}
