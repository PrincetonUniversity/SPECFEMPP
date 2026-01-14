#include "io/mesh/impl/fortran/dim2/read_parameters.hpp"
#include "io/fortranio/interface.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"

specfem::mesh::parameters<specfem::dimension::type::dim2>
specfem::io::mesh::impl::fortran::dim2::read_mesh_parameters(
    std::ifstream &stream) {
  // ---------------------------------------------------------------------
  // reading mesh properties

  int numat;           ///< Total number of different materials
  int ngnod;           ///< Number of control nodes
  int nspec;           ///< Number of spectral elements
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

  specfem::io::fortran_read_line(stream, &numat, &ngnod, &nspec, &pointsdisp,
                                 &plot_lowerleft_corner_only);

  // ---------------------------------------------------------------------
  // if (ngnod != 9) {
  //   std::ostringstream error_message;
  //   error_message << "Number of control nodes per element must be 9, but is "
  //                 << ngnod << "\n"
  //                 << "Currently, there is a bug when NGNOD == 4 \n";
  //   throw std::runtime_error(error_message.str());
  // }

  specfem::io::fortran_read_line(
      stream, &nelemabs, &nelem_acforcing, &nelem_acoustic_surface,
      &num_fluid_solid_edges, &num_fluid_poro_edges, &num_solid_poro_edges,
      &nnodes_tangential_curve, &nelem_on_the_axis);
  // ----------------------------------------------------------------------

  specfem::MPI::sync_all();

  return { numat,
           ngnod,
           nspec,
           pointsdisp,
           nelemabs,
           nelem_acforcing,
           nelem_acoustic_surface,
           num_fluid_solid_edges,
           num_fluid_poro_edges,
           num_solid_poro_edges,
           nnodes_tangential_curve,
           nelem_on_the_axis,
           plot_lowerleft_corner_only };
}
