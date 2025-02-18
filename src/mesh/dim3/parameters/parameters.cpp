#include "mesh/dim3/parameters/parameters.hpp"
#include <iostream>

void specfem::mesh::parameters<specfem::dimension::type::dim3>::print() const {
  std::cout << "Acoustic simulation:........................... "
            << acoustic_simulation << std::endl;
  std::cout << "Elastic simulation:............................ "
            << elastic_simulation << std::endl;
  std::cout << "Poroelastic simulation:........................ "
            << poroelastic_simulation << std::endl;
  std::cout << "Anisotropy:.................................... " << anisotropy
            << std::endl;
  std::cout << "Stacey ABC:.................................... " << stacey_abc
            << std::endl;
  std::cout << "PML ABC:....................................... " << pml_abc
            << std::endl;
  std::cout << "Approximate ocean load:........................ "
            << approximate_ocean_load << std::endl;
  std::cout << "Use mesh coloring:............................. "
            << use_mesh_coloring << std::endl;
  std::cout << "Number of dimensions:.......................... " << ndim
            << std::endl;
  std::cout << "Number of GLLX:................................ " << ngllx
            << std::endl;
  std::cout << "Number of GLLY:................................ " << nglly
            << std::endl;
  std::cout << "Number of GLLZ:................................ " << ngllz
            << std::endl;
  std::cout << "Number of GLL square:.......................... " << ngllsquare
            << std::endl;
  std::cout << "Number of processors:.......................... " << nproc
            << std::endl;
  std::cout << "Number of spectral elements:................... " << nspec
            << std::endl;
  std::cout << "Number of spectral elements for poroelastic:... " << nspec_poro
            << std::endl;
  std::cout << "Number of global nodes:........................ " << nglob
            << std::endl;
  std::cout << "Number of global nodes for ocean:.............. " << nglob_ocean
            << std::endl;
  std::cout << "Number of spectral elements for 2D bottom:..... "
            << nspec2D_bottom << std::endl;
  std::cout << "Number of spectral elements for 2D top:........ " << nspec2D_top
            << std::endl;
  std::cout << "Number of spectral elements for 2D xmin:....... "
            << nspec2D_xmin << std::endl;
  std::cout << "Number of spectral elements for 2D xmax:....... "
            << nspec2D_xmax << std::endl;
  std::cout << "Number of spectral elements for 2D ymin:....... "
            << nspec2D_ymin << std::endl;
  std::cout << "Number of spectral elements for 2D ymax:....... "
            << nspec2D_ymax << std::endl;
  std::cout << "Number of irregular spectral elements:......... "
            << nspec_irregular << std::endl;
  std::cout << "Number of neighbors:........................... "
            << num_neighbors << std::endl;
  std::cout << "Number of faces on the surface:................ "
            << nfaces_surface << std::endl;
  std::cout << "Number of absorbing boundary faces:............ "
            << num_abs_boundary_faces << std::endl;
  std::cout << "Number of free surface faces:.................. "
            << num_free_surface_faces << std::endl;
  std::cout << "Number of acoustic-elastic faces:.............. "
            << num_coupling_ac_el_faces << std::endl;
  std::cout << "Number of acoustic-poroelastic faces:.......... "
            << num_coupling_ac_po_faces << std::endl;
  std::cout << "Number of elastic-poroelastic faces:........... "
            << num_coupling_el_po_faces << std::endl;
  std::cout << "Number of poroelastic-elastic faces:........... "
            << num_coupling_po_el_faces << std::endl;
  std::cout << "Number of external mesh interfaces:............ "
            << num_interfaces_ext_mesh << std::endl;
  std::cout << "Maximum number of interfaces:.................. "
            << max_nibool_interfaces_ext_mesh << std::endl;
  std::cout << "Number of inner acoustic spectral elements:.... "
            << nspec_inner_acoustic << std::endl;
  std::cout << "Number of outer acoustic spectral elements:.... "
            << nspec_outer_acoustic << std::endl;
  std::cout << "Number of inner elastic spectral elements:..... "
            << nspec_inner_elastic << std::endl;
  std::cout << "Number of outer elastic spectral elements:..... "
            << nspec_outer_elastic << std::endl;
  std::cout << "Number of inner poroelastic spectral elements:. "
            << nspec_inner_poroelastic << std::endl;
  std::cout << "Number of outer poroelastic spectral elements:. "
            << nspec_outer_poroelastic << std::endl;
  std::cout << "Number of phase ispec acoustic: ............... "
            << num_phase_ispec_acoustic << std::endl;
  std::cout << "Number of phase ispec elastic:................. "
            << num_phase_ispec_elastic << std::endl;
  std::cout << "Number of phase ispec poroelastic:............. "
            << num_phase_ispec_poroelastic << std::endl;
  std::cout << "Number of colors inner acoustic:............... "
            << num_colors_inner_acoustic << std::endl;
  std::cout << "Number of colors outer acoustic:............... "
            << num_colors_outer_acoustic << std::endl;
  std::cout << "Number of colors inner elastic:................ "
            << num_colors_inner_elastic << std::endl;
  std::cout << "Number of colors outer elastic:................ "
            << num_colors_outer_elastic << std::endl;
}
