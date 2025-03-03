#include "mesh/mesh.hpp"
#include <iostream>

std::string
specfem::mesh::absorbing_boundary<specfem::dimension::type::dim3>::print()
    const {

  std::ostringstream message;

  // Print variables and metadata
  message << "Absorbing Boundary Metadata:"
          << "\n";
  message << "==============================================="
          << "\n";
  message << "  ngllsquare:............. " << ngllsquare << "\n";
  message << "  num_abs_boundary_faces:. " << num_abs_boundary_faces << "\n";
  message << "  acoustic:............... " << acoustic << "\n";
  message << "  elastic:................ " << elastic << "\n";
  message << "  nspec2D_xmin:........... " << nspec2D_xmin << "\n";
  message << "  nspec2D_xmax:........... " << nspec2D_xmax << "\n";
  message << "  nspec2D_ymin:........... " << nspec2D_ymin << "\n";
  message << "  nspec2D_ymax:........... " << nspec2D_ymax << "\n";
  message << "  NSPEC2D_BOTTOM:......... " << NSPEC2D_BOTTOM << "\n";
  message << "  NSPEC2D_TOP:............ " << NSPEC2D_TOP << "\n";

  // Print the absorbing ispec metadata
  message << "  Array sizes:"
          << "\n";
  message << "  -----------------------------------------------"
          << "\n";
  message << "  ispec:.................. " << ispec.extent(0) << "\n";
  message << "  ijk:.................... " << ijk.extent(0) << " "
          << ijk.extent(1) << " " << ijk.extent(2) << "\n";
  message << "  jacobian2Dw:............ " << jacobian2Dw.extent(0) << " "
          << jacobian2Dw.extent(1) << "\n";
  message << "  normal:................. " << normal.extent(0) << " "
          << normal.extent(1) << " " << normal.extent(2) << "\n";

  return message.str();
}

std::string
specfem::mesh::absorbing_boundary<specfem::dimension::type::dim3>::print_ijk(
    const int iface) const {
  std::string name = "ijk";
  std::ostringstream message;

  // Print the absorbing boundary metadata
  message << "Absorbing Boundary ijk for face " << iface << "\n";
  message << "===============================================\n";
  message << ""
          << "\n";
  message << " ---> igll squared\n"
          << "ijk:\n";

  // Print the ijk values
  for (int i = 0; i < 3; i++) {

    message << "ijk_boundary(" << name[i] << ") = ";
    for (int j = 0; j < ngllsquare; j++) {
      message << ijk(iface, i, j) << " ";
    };
    message << "\n";
  };

  return message.str();
};
