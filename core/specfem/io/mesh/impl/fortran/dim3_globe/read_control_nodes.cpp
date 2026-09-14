#include "specfem/io/mesh/impl/fortran/dim3_globe/read_control_nodes.hpp"

#include "specfem/io.hpp"
#include "specfem/io/fortranio/interface.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace specfem::io::mesh::impl::fortran::dim3_globe_impl {

void set_coordinate_bounds(
    specfem::mesh::control_nodes<specfem::element::dimension_tag::dim3>
        &nodes) {
  nodes.xmin = nodes.ymin = nodes.zmin = std::numeric_limits<type_real>::max();
  nodes.xmax = nodes.ymax = nodes.zmax =
      std::numeric_limits<type_real>::lowest();
  for (int inode = 0; inode < nodes.nnodes; ++inode) {
    nodes.xmin = std::min(nodes.xmin, nodes.coordinates(inode, 0));
    nodes.xmax = std::max(nodes.xmax, nodes.coordinates(inode, 0));
    nodes.ymin = std::min(nodes.ymin, nodes.coordinates(inode, 1));
    nodes.ymax = std::max(nodes.ymax, nodes.coordinates(inode, 1));
    nodes.zmin = std::min(nodes.zmin, nodes.coordinates(inode, 2));
    nodes.zmax = std::max(nodes.zmax, nodes.coordinates(inode, 2));
  }
}

} // namespace specfem::io::mesh::impl::fortran::dim3_globe_impl

int specfem::io::mesh::impl::fortran::dim3_globe::read_control_node_coordinates(
    std::ifstream &stream, specfem::mesh::globe3d_mesh &mesh, const int ngnod) {
  namespace reader_impl = specfem::io::mesh::impl::fortran::dim3_globe_impl;

  int nnode = 0;
  specfem::io::fortran_read_line(stream, &nnode);
  if (nnode <= 0) {
    throw std::runtime_error("Globe mesh database contains no anchor nodes");
  }

  mesh.control_nodes = { ngnod, nnode };
  std::vector<double> x(nnode), y(nnode), z(nnode);
  specfem::io::fortran_read_line(stream, &x, &y, &z);

  auto coordinates = mesh.control_nodes.coordinates;
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      x_view(x.data(), nnode), y_view(y.data(), nnode), z_view(z.data(), nnode);
  Kokkos::parallel_for(
      "specfem::io::mesh::dim3_globe::read_control_nodes::coordinates",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nnode),
      [=](const int inode) {
        coordinates(inode, 0) = x_view(inode);
        coordinates(inode, 1) = y_view(inode);
        coordinates(inode, 2) = z_view(inode);
      });
  Kokkos::fence();
  reader_impl::set_coordinate_bounds(mesh.control_nodes);

  auto &globe = mesh.globe;
  globe.reference_coordinates =
      specfem::mesh::globe_mesh_data::CoordinatesViewType(
          "specfem::mesh::globe_reference_coordinates", nnode);
  if (globe.has_reference_geometry) {
    specfem::io::fortran_read_line(stream, &x, &y, &z);
  }

  const bool has_reference_geometry = globe.has_reference_geometry;
  auto reference_coordinates = globe.reference_coordinates;
  Kokkos::parallel_for(
      "specfem::io::mesh::dim3_globe::read_control_nodes::reference_"
      "coordinates",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nnode),
      [=](const int inode) {
        reference_coordinates(inode, 0) =
            has_reference_geometry ? x_view(inode) : coordinates(inode, 0);
        reference_coordinates(inode, 1) =
            has_reference_geometry ? y_view(inode) : coordinates(inode, 1);
        reference_coordinates(inode, 2) =
            has_reference_geometry ? z_view(inode) : coordinates(inode, 2);
      });
  Kokkos::fence();

  return nnode;
}

void specfem::io::mesh::impl::fortran::dim3_globe::read_control_node_indices(
    std::ifstream &stream, specfem::mesh::globe3d_mesh &mesh, const int ngnod,
    const int nnode) {
  std::vector<int> node_ids(static_cast<std::size_t>(ngnod) * mesh.nspec);
  specfem::io::fortran_read_line(stream, &node_ids);
  mesh.control_nodes.control_node_index =
      Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace>(
          "specfem::mesh::globe_control_node_index", mesh.nspec, ngnod);

  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      node_ids_view(node_ids.data(), node_ids.size());
  auto control_node_index = mesh.control_nodes.control_node_index;
  int invalid_anchor_count = 0;
  Kokkos::parallel_reduce(
      "specfem::io::mesh::dim3_globe::read_control_nodes::control_node_index",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
          { 0, 0 }, { mesh.nspec, ngnod }),
      [=](const int ispec, const int globe_anchor,
          int &local_invalid_anchor_count) {
        const int inode = node_ids_view(ispec * ngnod + globe_anchor) - 1;
        if (inode < 0 || inode >= nnode) {
          ++local_invalid_anchor_count;
          return;
        }
        control_node_index(ispec, globe_anchor) = inode;
      },
      invalid_anchor_count);
  if (invalid_anchor_count > 0) {
    throw std::runtime_error("Invalid anchor node ID in globe database");
  }
}
