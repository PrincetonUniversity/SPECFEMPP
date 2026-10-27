#pragma once

#include "injection_surface.hpp"
#include "specfem/assembly/boundaries/dim3/impl/utilities.hpp"
#include "specfem/point.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem::assembly::injection_surface_impl {

/** @brief An injection face on a compute element. */
struct Face {
  int ispec;                             ///< Compute element
  specfem::mesh_entity::dim3::type type; ///< Face of the element
  int side;                              ///< +1 or -1, see InjectionSurface
};

/** @brief Whether @p type is one of the six faces of an element. */
bool is_face(specfem::mesh_entity::dim3::type type);

} // namespace specfem::assembly::injection_surface_impl

template <specfem::simulation::model ModelTag>
specfem::assembly::InjectionSurface<specfem::element::dimension_tag::dim3>::
    InjectionSurface(
        const specfem::mesh::mesh<ModelTag> &mesh,
        const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::element_types<dimension_tag> &element_types,
        const specfem::assembly::boundaries<dimension_tag> &boundaries) {
  using Face = specfem::assembly::injection_surface_impl::Face;
  const auto &surface = mesh.boundaries.injection_surface;
  const int ngllz = mesh_assembly.element_grid.ngllz;
  const int nglly = mesh_assembly.element_grid.nglly;
  const int ngllx = mesh_assembly.element_grid.ngllx;

  std::vector<Face> faces;
  for (int i = 0; i < surface.nfaces; ++i) {
    const auto type = surface.type(i);
    if (!specfem::assembly::injection_surface_impl::is_face(type))
      throw std::runtime_error("Injection face " + std::to_string(i) +
                               " is not a whole element face.");
    faces.push_back({ mesh_assembly.h_mesh_to_compute(surface.index_mapping(i)),
                      type, surface.side(i) });
  }

  const auto points_on = [&](const Face &face) {
    std::vector<std::array<int, 3>> points;
    for (int iz = 0; iz < ngllz; ++iz)
      for (int iy = 0; iy < nglly; ++iy)
        for (int ix = 0; ix < ngllx; ++ix)
          if (specfem::assembly::boundaries_impl::is_on_boundary(
                  face.type, iz, iy, ix, ngllz, nglly, ngllx))
            points.push_back({ iz, iy, ix });
    return points;
  };

  // Faces of each material system, in group order.
  std::vector<std::vector<Face>> faces_by_system;
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    auto &bucket = faces_by_system.emplace_back();
    for (const auto &face : faces)
      if (element_types.get_medium_tag(face.ispec) == TagsType::medium_tag &&
          element_types.get_property_tag(face.ispec) == TagsType::property_tag)
        bucket.push_back(face);
  });

  for (const auto &bucket : faces_by_system)
    for (const auto &face : bucket)
      nentries += static_cast<int>(points_on(face).size());
  nrows = nentries;

  ispec = IndexView("specfem::assembly::injection_surface::ispec", nentries);
  iz = IndexView("specfem::assembly::injection_surface::iz", nentries);
  iy = IndexView("specfem::assembly::injection_surface::iy", nentries);
  ix = IndexView("specfem::assembly::injection_surface::ix", nentries);
  row = IndexView("specfem::assembly::injection_surface::row", nentries);
  normal = NormalView("specfem::assembly::injection_surface::normal", nentries);
  weight = WeightView("specfem::assembly::injection_surface::weight", nentries);
  boundary_tag = BoundaryTagView(
      "specfem::assembly::injection_surface::boundary_tag", nentries);
  h_ispec = Kokkos::create_mirror_view(ispec);
  h_iz = Kokkos::create_mirror_view(iz);
  h_iy = Kokkos::create_mirror_view(iy);
  h_ix = Kokkos::create_mirror_view(ix);
  h_row = Kokkos::create_mirror_view(row);
  h_normal = Kokkos::create_mirror_view(normal);
  h_weight = Kokkos::create_mirror_view(weight);
  h_boundary_tag = Kokkos::create_mirror_view(boundary_tag);

  x = RowView("specfem::assembly::injection_surface::x", nrows);
  y = RowView("specfem::assembly::injection_surface::y", nrows);
  z = RowView("specfem::assembly::injection_surface::z", nrows);
  nx = RowView("specfem::assembly::injection_surface::nx", nrows);
  ny = RowView("specfem::assembly::injection_surface::ny", nrows);
  nz = RowView("specfem::assembly::injection_surface::nz", nrows);
  h_row_entry = Kokkos::View<int *, Kokkos::HostSpace>(
      "specfem::assembly::injection_surface::row_entry", nrows);

  int entry = 0;
  std::size_t system = 0;
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    const int first = entry;
    for (const auto &face : faces_by_system[system])
      for (const auto &[jz, jy, jx] : points_on(face)) {
        specfem::point::index<dimension_tag> index(face.ispec, jz, jy, jx);
        specfem::point::jacobian_matrix<dimension_tag, true, false> jacobian;
        specfem::assembly::load_on_host(index, jacobian_matrix, jacobian);
        const std::array<type_real, 3> gll_weights = {
          mesh_assembly.h_weights(jz), mesh_assembly.h_weights(jy),
          mesh_assembly.h_weights(jx)
        };
        const auto [n, w] =
            specfem::assembly::boundaries_impl::get_boundary_face_and_weight(
                face.type, gll_weights, jacobian);
        const type_real length =
            std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (!(length > 0))
          throw std::runtime_error("Injection face of element " +
                                   std::to_string(face.ispec) +
                                   " has a degenerate normal.");

        h_ispec(entry) = face.ispec;
        h_iz(entry) = jz;
        h_iy(entry) = jy;
        h_ix(entry) = jx;
        h_row(entry) = entry;
        for (int d = 0; d < 3; ++d)
          h_normal(entry, d) = face.side * n[d] / length;
        h_weight(entry) = w * length;
        h_boundary_tag(entry) =
            boundaries.get_boundary_tag_on_host(face.ispec, jz, jy, jx);

        x(entry) = mesh_assembly.h_coord(face.ispec, jz, jy, jx, 0);
        y(entry) = mesh_assembly.h_coord(face.ispec, jz, jy, jx, 1);
        z(entry) = mesh_assembly.h_coord(face.ispec, jz, jy, jx, 2);
        nx(entry) = h_normal(entry, 0);
        ny(entry) = h_normal(entry, 1);
        nz(entry) = h_normal(entry, 2);
        h_row_entry(entry) = entry;
        ++entry;
      }
    entry_ranges.template get<TagsType>() = { first, entry };
    row_ranges.template get<TagsType>() = { first, entry };
    ++system;
  });

  Kokkos::deep_copy(ispec, h_ispec);
  Kokkos::deep_copy(iz, h_iz);
  Kokkos::deep_copy(iy, h_iy);
  Kokkos::deep_copy(ix, h_ix);
  Kokkos::deep_copy(row, h_row);
  Kokkos::deep_copy(normal, h_normal);
  Kokkos::deep_copy(weight, h_weight);
  Kokkos::deep_copy(boundary_tag, h_boundary_tag);
}
