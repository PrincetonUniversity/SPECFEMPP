#include "IO/fortranio/interface.hpp"
#include "IO/mesh/impl/fortran/dim3/interface.hpp"
#include "enumerations/dimension.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"

specfem::mesh::parameters<specfem::dimension::type::dim3>
specfem::IO::mesh::impl::fortran::dim3::read_mesh_parameters(
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  // Initialize test parameter
  int itest = -9999;

  // Read testparamater
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9999) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  }

  // Initialize flags
  bool acoustic_simulation;
  bool elastic_simulation;
  bool poroelastic_simulation;
  bool anisotropy;
  bool stacey_abc;
  bool pml_abc;
  bool approximate_ocean_load;
  bool use_mesh_coloring;

  // Read parameters
  specfem::IO::fortran_read_line(stream, &acoustic_simulation);
  specfem::IO::fortran_read_line(stream, &elastic_simulation);
  specfem::IO::fortran_read_line(stream, &poroelastic_simulation);
  specfem::IO::fortran_read_line(stream, &anisotropy);
  specfem::IO::fortran_read_line(stream, &stacey_abc);
  specfem::IO::fortran_read_line(stream, &pml_abc);
  specfem::IO::fortran_read_line(stream, &approximate_ocean_load);
  specfem::IO::fortran_read_line(stream, &use_mesh_coloring);

#ifndef NDEBUG
  // Print the read parameters
  std::cout << "acoustic_simulation:...." << acoustic_simulation << std::endl;
  std::cout << "elastic_simulation:....." << elastic_simulation << std::endl;
  std::cout << "poroelastic_simulation:." << poroelastic_simulation
            << std::endl;
  std::cout << "anisotropy:............." << anisotropy << std::endl;
  std::cout << "stacey_abc:............." << stacey_abc << std::endl;
  std::cout << "pml_abc:................" << pml_abc << std::endl;
  std::cout << "approximate_ocean_load:." << approximate_ocean_load
            << std::endl;
  std::cout << "use_mesh_coloring:......" << use_mesh_coloring << std::endl;
#endif

  // Read test parameter
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9998) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  };

  // Initialize first bout of integer parameters
  int ndim;
  int ngllx;
  int nglly;
  int ngllz;
  int ngllsquare;
  int nproc;

  // Read parameters
  specfem::IO::fortran_read_line(stream, &ndim);
  specfem::IO::fortran_read_line(stream, &ngllx);
  specfem::IO::fortran_read_line(stream, &nglly);
  specfem::IO::fortran_read_line(stream, &ngllz);
  specfem::IO::fortran_read_line(stream, &ngllsquare);
  specfem::IO::fortran_read_line(stream, &nproc);

#ifndef NDEBUG
  // Print the read parameters
  std::cout << "ndim:................." << ndim << std::endl;
  std::cout << "ngllx:................" << ngllx << std::endl;
  std::cout << "ngly:................." << nglly << std::endl;
  std::cout << "ngllz:................" << ngllz << std::endl;
  std::cout << "ngllsquare:..........." << ngllsquare << std::endl;
  std::cout << "nproc:................" << nproc << std::endl;
#endif

  // Read test parameter
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9997) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  };

  // Initialize second bout of integer parameters
  int nspec;
  int nspec_poro;
  int nglob;
  int nglob_ocean;

  // Read parameters
  specfem::IO::fortran_read_line(stream, &nspec);
  specfem::IO::fortran_read_line(stream, &nspec_poro);
  specfem::IO::fortran_read_line(stream, &nglob);
  specfem::IO::fortran_read_line(stream, &nglob_ocean);

#ifndef NDEBUG
  // Print the read parameters
  std::cout << "nspec:................." << nspec << std::endl;
  std::cout << "nspec_poro:............" << nspec_poro << std::endl;
  std::cout << "nglob:................." << nglob << std::endl;
  std::cout << "nglob_ocean:............" << nglob_ocean << std::endl;
#endif

  // Read test parameter
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9996) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  };

  int nspec2D_bottom;
  int nspec2D_top;
  int nspec2D_xmin;
  int nspec2D_xmax;
  int nspec2D_ymin;
  int nspec2D_ymax;
  int nspec_irregular;

  specfem::IO::fortran_read_line(stream, &nspec2D_bottom);
  specfem::IO::fortran_read_line(stream, &nspec2D_top);
  specfem::IO::fortran_read_line(stream, &nspec2D_xmin);
  specfem::IO::fortran_read_line(stream, &nspec2D_xmax);
  specfem::IO::fortran_read_line(stream, &nspec2D_ymin);
  specfem::IO::fortran_read_line(stream, &nspec2D_ymax);
  specfem::IO::fortran_read_line(stream, &nspec_irregular);

#ifndef NDEBUG
  std::cout << "nspec2D_bottom:........" << nspec2D_bottom << std::endl;
  std::cout << "nspec2D_top:..........." << nspec2D_top << std::endl;
  std::cout << "nspec2D_xmin:.........." << nspec2D_xmin << std::endl;
  std::cout << "nspec2D_xmax:.........." << nspec2D_xmax << std::endl;
  std::cout << "nspec2D_ymin:.........." << nspec2D_ymin << std::endl;
  std::cout << "nspec2D_ymax:.........." << nspec2D_ymax << std::endl;
  std::cout << "nspec_irregular:......." << nspec_irregular << std::endl;
#endif

  // Read test parameter
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9995) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  };

  int num_neighbors;
  int nfaces_surface;
  int num_abs_boundary_faces;
  int num_free_surface_faces;
  int num_coupling_ac_el_faces;
  int num_coupling_ac_po_faces;
  int num_coupling_el_po_faces;
  int num_coupling_po_el_faces;
  int num_interfaces_ext_mesh;
  int max_nibool_interfaces_ext_mesh;

  specfem::IO::fortran_read_line(stream, &num_neighbors);
  specfem::IO::fortran_read_line(stream, &nfaces_surface);
  specfem::IO::fortran_read_line(stream, &num_abs_boundary_faces);
  specfem::IO::fortran_read_line(stream, &num_free_surface_faces);
  specfem::IO::fortran_read_line(stream, &num_coupling_ac_el_faces);
  specfem::IO::fortran_read_line(stream, &num_coupling_ac_po_faces);
  specfem::IO::fortran_read_line(stream, &num_coupling_el_po_faces);
  specfem::IO::fortran_read_line(stream, &num_coupling_po_el_faces);
  specfem::IO::fortran_read_line(stream, &num_interfaces_ext_mesh);
  specfem::IO::fortran_read_line(stream, &max_nibool_interfaces_ext_mesh);

#ifndef NDEBUG
  std::cout << "num_neighbors:.........." << num_neighbors << std::endl;
  std::cout << "nfaces_surface:........." << nfaces_surface << std::endl;
  std::cout << "num_abs_boundary_faces:." << num_abs_boundary_faces
            << std::endl;
  std::cout << "num_free_surface_faces:." << num_free_surface_faces
            << std::endl;
  std::cout << "num_coupling_ac_el_faces:" << num_coupling_ac_el_faces
            << std::endl;
  std::cout << "num_coupling_ac_po_faces:" << num_coupling_ac_po_faces
            << std::endl;
  std::cout << "num_coupling_el_po_faces:" << num_coupling_el_po_faces
            << std::endl;
  std::cout << "num_coupling_po_el_faces:" << num_coupling_po_el_faces
            << std::endl;
  std::cout << "num_interfaces_ext_mesh:." << num_interfaces_ext_mesh
            << std::endl;
  std::cout << "max_nibool_interfaces_ext_mesh:"
            << max_nibool_interfaces_ext_mesh << std::endl;
#endif

  // Read test parameter
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9994) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  };

  int nspec_inner_acoustic;
  int nspec_outer_acoustic;
  int nspec_inner_elastic;
  int nspec_outer_elastic;
  int nspec_inner_poroelastic;
  int nspec_outer_poroelastic;

  specfem::IO::fortran_read_line(stream, &nspec_inner_acoustic);
  specfem::IO::fortran_read_line(stream, &nspec_outer_acoustic);
  specfem::IO::fortran_read_line(stream, &nspec_inner_elastic);
  specfem::IO::fortran_read_line(stream, &nspec_outer_elastic);
  specfem::IO::fortran_read_line(stream, &nspec_inner_poroelastic);
  specfem::IO::fortran_read_line(stream, &nspec_outer_poroelastic);

#ifndef NDEBUG
  std::cout << "nspec_inner_acoustic:..." << nspec_inner_acoustic << std::endl;
  std::cout << "nspec_outer_acoustic:..." << nspec_outer_acoustic << std::endl;
  std::cout << "nspec_inner_elastic:...." << nspec_inner_elastic << std::endl;
  std::cout << "nspec_outer_elastic:...." << nspec_outer_elastic << std::endl;
  std::cout << "nspec_inner_poroelastic:." << nspec_inner_poroelastic
            << std::endl;
  std::cout << "nspec_outer_poroelastic:." << nspec_outer_poroelastic
            << std::endl;
#endif

  // Read test parameter
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9993) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  };

  int num_phase_ispec_acoustic;
  int num_phase_ispec_elastic;
  int num_phase_ispec_poroelastic;
  int num_colors_inner_acoustic;
  int num_colors_outer_acoustic;
  int num_colors_inner_elastic;
  int num_colors_outer_elastic;

  specfem::IO::fortran_read_line(stream, &num_phase_ispec_acoustic);
  specfem::IO::fortran_read_line(stream, &num_phase_ispec_elastic);
  specfem::IO::fortran_read_line(stream, &num_phase_ispec_poroelastic);
  specfem::IO::fortran_read_line(stream, &num_colors_inner_acoustic);
  specfem::IO::fortran_read_line(stream, &num_colors_outer_acoustic);
  specfem::IO::fortran_read_line(stream, &num_colors_inner_elastic);
  specfem::IO::fortran_read_line(stream, &num_colors_outer_elastic);

#ifndef NDEBUG
  std::cout << "num_phase_ispec_acoustic:" << num_phase_ispec_acoustic
            << std::endl;
  std::cout << "num_phase_ispec_elastic:." << num_phase_ispec_elastic
            << std::endl;
  std::cout << "num_phase_ispec_poroelastic:" << num_phase_ispec_poroelastic
            << std::endl;
  std::cout << "num_colors_inner_acoustic:" << num_colors_inner_acoustic
            << std::endl;
  std::cout << "num_colors_outer_acoustic:" << num_colors_outer_acoustic
            << std::endl;
  std::cout << "num_colors_inner_elastic:" << num_colors_inner_elastic
            << std::endl;
  std::cout << "num_colors_outer_elastic:" << num_colors_outer_elastic
            << std::endl;
#endif

  // Read test parameter
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9992) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  };

  mpi->sync_all();

  return { acoustic_simulation,
           elastic_simulation,
           poroelastic_simulation,
           anisotropy,
           stacey_abc,
           pml_abc,
           approximate_ocean_load,
           use_mesh_coloring,
           ndim,
           ngllx,
           nglly,
           ngllz,
           ngllsquare,
           nproc,
           nspec,
           nspec_poro,
           nglob,
           nglob_ocean,
           nspec2D_bottom,
           nspec2D_top,
           nspec2D_xmin,
           nspec2D_xmax,
           nspec2D_ymin,
           nspec2D_ymax,
           nspec_irregular,
           num_neighbors,
           nfaces_surface,
           num_abs_boundary_faces,
           num_free_surface_faces,
           num_coupling_ac_el_faces,
           num_coupling_ac_po_faces,
           num_coupling_el_po_faces,
           num_coupling_po_el_faces,
           num_interfaces_ext_mesh,
           max_nibool_interfaces_ext_mesh,
           nspec_inner_acoustic,
           nspec_outer_acoustic,
           nspec_inner_elastic,
           nspec_outer_elastic,
           nspec_inner_poroelastic,
           nspec_outer_poroelastic,
           num_phase_ispec_acoustic,
           num_phase_ispec_elastic,
           num_phase_ispec_poroelastic,
           num_colors_inner_acoustic,
           num_colors_outer_acoustic,
           num_colors_inner_elastic,
           num_colors_outer_elastic };
}
