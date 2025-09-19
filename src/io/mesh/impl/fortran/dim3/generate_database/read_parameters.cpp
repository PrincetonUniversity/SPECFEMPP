#include "enumerations/dimension.hpp"
#include "io/fortranio/interface.hpp"
#include "io/mesh/impl/fortran/dim3/generate_database/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"

specfem::mesh::parameters<specfem::dimension::type::dim3>
specfem::io::mesh::impl::fortran::dim3::read_mesh_parameters(
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  // Creating aliases for Array Reading functions
  using specfem::io::mesh::impl::fortran::dim3::check_read_test_value;
  using specfem::io::mesh::impl::fortran::dim3::try_read_line;

  // Initialize test parameter
  check_read_test_value(stream, 9999);

  // Initialize parameters object
  specfem::mesh::parameters<specfem::dimension::type::dim3> parameters;

  // Read parameters
  try_read_line("parameters_acoustic_simulation", stream,
                &parameters.acoustic_simulation);
  try_read_line("parameters_elastic_simulation", stream,
                &parameters.elastic_simulation);
  try_read_line("parameters_poroelastic_simulation", stream,
                &parameters.poroelastic_simulation);
  try_read_line("parameters_anisotropy", stream, &parameters.anisotropy);
  try_read_line("parameters_stacey_abc", stream, &parameters.stacey_abc);
  try_read_line("parameters_pml_abc", stream, &parameters.pml_abc);
  try_read_line("parameters_approximate_ocean_load", stream,
                &parameters.approximate_ocean_load);
  try_read_line("parameters_use_mesh_coloring", stream,
                &parameters.use_mesh_coloring);

#ifndef NDEBUG
  // Print the read parameters
  std::cout << "acoustic_simulation:...." << parameters.acoustic_simulation
            << std::endl;
  std::cout << "elastic_simulation:....." << parameters.elastic_simulation
            << std::endl;
  std::cout << "poroelastic_simulation:." << parameters.poroelastic_simulation
            << std::endl;
  std::cout << "anisotropy:............." << parameters.anisotropy << std::endl;
  std::cout << "stacey_abc:............." << parameters.stacey_abc << std::endl;
  std::cout << "pml_abc:................" << parameters.pml_abc << std::endl;
  std::cout << "approximate_ocean_load:." << parameters.approximate_ocean_load
            << std::endl;
  std::cout << "use_mesh_coloring:......" << parameters.use_mesh_coloring
            << std::endl;
#endif

  // Read test parameter
  check_read_test_value(stream, 9998);

  // Read parameters
  try_read_line("mesh.parameters.ndim", stream, &parameters.ndim);
  try_read_line("mesh.parameters.ngllx", stream, &parameters.ngllx);
  try_read_line("mesh.parameters.nglly", stream, &parameters.nglly);
  try_read_line("mesh.parameters.ngllz", stream, &parameters.ngllz);
  try_read_line("mesh.parameters.ngllsquare", stream, &parameters.ngllsquare);
  try_read_line("mesh.parameters.ngnod", stream, &parameters.ngnod);
  try_read_line("mesh.parameters.nproc", stream, &parameters.nproc);

#ifndef NDEBUG
  // Print the read parameters
  std::cout << "ndim:................." << parameters.ndim << std::endl;
  std::cout << "ngllx:................" << parameters.ngllx << std::endl;
  std::cout << "ngly:................." << parameters.nglly << std::endl;
  std::cout << "ngllz:................" << parameters.ngllz << std::endl;
  std::cout << "ngllsquare:..........." << parameters.ngllsquare << std::endl;
  std::cout << "nproc:................" << parameters.nproc << std::endl;
#endif

  // Read test parameter
  check_read_test_value(stream, 9997);

  // Read parameters
  try_read_line("mesh.parameters.nspec", stream, &parameters.nspec);
  try_read_line("mesh.parameters.nspec_poro", stream, &parameters.nspec_poro);
  try_read_line("mesh.parameters.nglob", stream, &parameters.nglob);
  try_read_line("mesh.parameters.nglob_ocean", stream, &parameters.nglob_ocean);

#ifndef NDEBUG
  // Print the read parameters
  std::cout << "nspec:................." << parameters.nspec << std::endl;
  std::cout << "nspec_poro:............" << parameters.nspec_poro << std::endl;
  std::cout << "nglob:................." << parameters.nglob << std::endl;
  std::cout << "nglob_ocean:............" << parameters.nglob_ocean
            << std::endl;
#endif

  // Read test parameter
  check_read_test_value(stream, 9996);

  // Read parameters
  try_read_line("mesh.parameters.nspec2D_bottom", stream,
                &parameters.nspec2D_bottom);
  try_read_line("mesh.parameters.nspec2D_top", stream, &parameters.nspec2D_top);
  try_read_line("mesh.parameters.nspec2D_xmin", stream,
                &parameters.nspec2D_xmin);
  try_read_line("mesh.parameters.nspec2D_xmax", stream,
                &parameters.nspec2D_xmax);
  try_read_line("mesh.parameters.nspec2D_ymin", stream,
                &parameters.nspec2D_ymin);
  try_read_line("mesh.parameters.nspec2D_ymax", stream,
                &parameters.nspec2D_ymax);
  try_read_line("mesh.parameters.nspec_irregular", stream,
                &parameters.nspec_irregular);
  try_read_line("mesh.parameters.nnodes_ext_mesh", stream, &parameters.nnodes);

#ifndef NDEBUG
  std::cout << "nspec2D_bottom:........" << parameters.nspec2D_bottom
            << std::endl;
  std::cout << "nspec2D_top:..........." << parameters.nspec2D_top << std::endl;
  std::cout << "nspec2D_xmin:.........." << parameters.nspec2D_xmin
            << std::endl;
  std::cout << "nspec2D_xmax:.........." << parameters.nspec2D_xmax
            << std::endl;
  std::cout << "nspec2D_ymin:.........." << parameters.nspec2D_ymin
            << std::endl;
  std::cout << "nspec2D_ymax:.........." << parameters.nspec2D_ymax
            << std::endl;
  std::cout << "nspec_irregular:......." << parameters.nspec_irregular
            << std::endl;
#endif

  // Read test parameter
  check_read_test_value(stream, 9995);

  // Read parameters
  try_read_line("mesh.parameters.num_neighbors", stream,
                &parameters.num_neighbors);
  try_read_line("mesh.parameters.nfaces_surface", stream,
                &parameters.nfaces_surface);
  try_read_line("mesh.parameters.num_abs_boundary_faces", stream,
                &parameters.num_abs_boundary_faces);
  try_read_line("mesh.parameters.num_free_surface_faces", stream,
                &parameters.num_free_surface_faces);
  try_read_line("mesh.parameters.num_coupling_ac_el_faces", stream,
                &parameters.num_coupling_ac_el_faces);
  try_read_line("mesh.parameters.num_coupling_ac_po_faces", stream,
                &parameters.num_coupling_ac_po_faces);
  try_read_line("mesh.parameters.num_coupling_el_po_faces", stream,
                &parameters.num_coupling_el_po_faces);
  try_read_line("mesh.parameters.num_coupling_po_el_faces", stream,
                &parameters.num_coupling_po_el_faces);
  try_read_line("mesh.parameters.num_interfaces_ext_mesh", stream,
                &parameters.num_interfaces_ext_mesh);
  try_read_line("mesh.parameters.max_nibool_interfaces_ext_mesh", stream,
                &parameters.max_nibool_interfaces_ext_mesh);

#ifndef NDEBUG
  std::cout << "num_neighbors:.........." << parameters.num_neighbors
            << std::endl;
  std::cout << "nfaces_surface:........." << parameters.nfaces_surface
            << std::endl;
  std::cout << "num_abs_boundary_faces:." << parameters.num_abs_boundary_faces
            << std::endl;
  std::cout << "num_free_surface_faces:." << parameters.num_free_surface_faces
            << std::endl;
  std::cout << "num_coupling_ac_el_faces:"
            << parameters.num_coupling_ac_el_faces << std::endl;
  std::cout << "num_coupling_ac_po_faces:"
            << parameters.num_coupling_ac_po_faces << std::endl;
  std::cout << "num_coupling_el_po_faces:"
            << parameters.num_coupling_el_po_faces << std::endl;
  std::cout << "num_coupling_po_el_faces:"
            << parameters.num_coupling_po_el_faces << std::endl;
  std::cout << "num_interfaces_ext_mesh:." << parameters.num_interfaces_ext_mesh
            << std::endl;
  std::cout << "max_nibool_interfaces_ext_mesh:"
            << parameters.max_nibool_interfaces_ext_mesh << std::endl;
#endif

  // Read test parameter
  check_read_test_value(stream, 9994);

  // Read parameters
  try_read_line("mesh.parameters.nspec_inner_acoustic", stream,
                &parameters.nspec_inner_acoustic);
  try_read_line("mesh.parameters.nspec_outer_acoustic", stream,
                &parameters.nspec_outer_acoustic);
  try_read_line("mesh.parameters.nspec_inner_elastic", stream,
                &parameters.nspec_inner_elastic);
  try_read_line("mesh.parameters.nspec_outer_elastic", stream,
                &parameters.nspec_outer_elastic);
  try_read_line("mesh.parameters.nspec_inner_poroelastic", stream,
                &parameters.nspec_inner_poroelastic);
  try_read_line("mesh.parameters.nspec_outer_poroelastic", stream,
                &parameters.nspec_outer_poroelastic);

#ifndef NDEBUG
  std::cout << "nspec_inner_acoustic:..." << parameters.nspec_inner_acoustic
            << std::endl;
  std::cout << "nspec_outer_acoustic:..." << parameters.nspec_outer_acoustic
            << std::endl;
  std::cout << "nspec_inner_elastic:...." << parameters.nspec_inner_elastic
            << std::endl;
  std::cout << "nspec_outer_elastic:...." << parameters.nspec_outer_elastic
            << std::endl;
  std::cout << "nspec_inner_poroelastic:." << parameters.nspec_inner_poroelastic
            << std::endl;
  std::cout << "nspec_outer_poroelastic:." << parameters.nspec_outer_poroelastic
            << std::endl;
#endif

  // Read test parameter
  check_read_test_value(stream, 9993);

  // Read parameters
  try_read_line("mesh.parameters.num_phase_ispec_acoustic", stream,
                &parameters.num_phase_ispec_acoustic);
  try_read_line("mesh.parameters.num_phase_ispec_elastic", stream,
                &parameters.num_phase_ispec_elastic);
  try_read_line("mesh.parameters.num_phase_ispec_poroelastic", stream,
                &parameters.num_phase_ispec_poroelastic);
  try_read_line("mesh.parameters.num_colors_inner_acoustic", stream,
                &parameters.num_colors_inner_acoustic);
  try_read_line("mesh.parameters.num_colors_outer_acoustic", stream,
                &parameters.num_colors_outer_acoustic);
  try_read_line("mesh.parameters.num_colors_inner_elastic", stream,
                &parameters.num_colors_inner_elastic);
  try_read_line("mesh.parameters.num_colors_outer_elastic", stream,
                &parameters.num_colors_outer_elastic);

#ifndef NDEBUG
  std::cout << "num_phase_ispec_acoustic:"
            << parameters.num_phase_ispec_acoustic << std::endl;
  std::cout << "num_phase_ispec_elastic:." << parameters.num_phase_ispec_elastic
            << std::endl;
  std::cout << "num_phase_ispec_poroelastic:"
            << parameters.num_phase_ispec_poroelastic << std::endl;
  std::cout << "num_colors_inner_acoustic:"
            << parameters.num_colors_inner_acoustic << std::endl;
  std::cout << "num_colors_outer_acoustic:"
            << parameters.num_colors_outer_acoustic << std::endl;
  std::cout << "num_colors_inner_elastic:"
            << parameters.num_colors_inner_elastic << std::endl;
  std::cout << "num_colors_outer_elastic:"
            << parameters.num_colors_outer_elastic << std::endl;
#endif

  /// Read test parameter
  check_read_test_value(stream, 9992);

  mpi->sync_all();

  return parameters;
}
