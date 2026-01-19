
#include "io/mesh/impl/fortran/dim3/read_control_nodes.hpp"
#include "io/fortranio/interface.hpp"
#include "specfem/mesh.hpp"
#include <fstream>

specfem::mesh::control_nodes<specfem::dimension::type::dim3>
specfem::io::mesh::impl::fortran::dim3::read_control_nodes(
    std::ifstream &stream) {

  using ControlNodesType =
      specfem::mesh::control_nodes<specfem::dimension::type::dim3>;

  int ngnod;
  specfem::io::fortran_read_line(stream, &ngnod);

  int nnodes;
  specfem::io::fortran_read_line(stream, &nnodes);
  ControlNodesType control_nodes(ngnod, nnodes);

  // get maxima and minima of coordinates
  double x_min = std::numeric_limits<double>::max(), y_min = x_min,
         z_min = x_min;
  double x_max = std::numeric_limits<double>::lowest(), y_max = x_max,
         z_max = x_max;

  // Read control nodes coordinates one by one
  for (int inode = 0; inode < nnodes; ++inode) {
    int index;
    double x, y, z;
    specfem::io::fortran_read_line(stream, &index, &x, &y, &z);
    control_nodes.coordinates(index - 1, 0) = x;
    control_nodes.coordinates(index - 1, 1) = y;
    control_nodes.coordinates(index - 1, 2) = z;
    if (x < x_min)
      x_min = x;
    if (x > x_max)
      x_max = x;
    if (y < y_min)
      y_min = y;
    if (y > y_max)
      y_max = y;
    if (z < z_min)
      z_min = z;
    if (z > z_max)
      z_max = z;
  }

  control_nodes.xmin = x_min;
  control_nodes.xmax = x_max;
  control_nodes.ymin = y_min;
  control_nodes.ymax = y_max;
  control_nodes.zmin = z_min;
  control_nodes.zmax = z_max;

  return control_nodes;
}
