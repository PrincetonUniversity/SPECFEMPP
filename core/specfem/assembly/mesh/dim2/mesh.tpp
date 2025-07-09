#pragma once

#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "jacobian/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

namespace {
struct qp {
  type_real x = 0, z = 0;
  int iloc = 0, iglob = 0;
};

type_real get_tolerance(std::vector<qp> cart_cord, const int nspec,
                        const int ngllxz) {

  assert(cart_cord.size() == ngllxz * nspec);

  type_real xtypdist = std::numeric_limits<type_real>::max();
  for (int ispec = 0; ispec < nspec; ispec++) {
    type_real xmax = std::numeric_limits<type_real>::min();
    type_real xmin = std::numeric_limits<type_real>::max();
    type_real ymax = std::numeric_limits<type_real>::min();
    type_real ymin = std::numeric_limits<type_real>::max();
    for (int xz = 0; xz < ngllxz; xz++) {
      int iloc = ispec * (ngllxz) + xz;
      xmax = std::max(xmax, cart_cord[iloc].x);
      xmin = std::min(xmin, cart_cord[iloc].x);
      ymax = std::max(ymax, cart_cord[iloc].z);
      ymin = std::min(ymin, cart_cord[iloc].z);
    }

    xtypdist = std::min(xtypdist, xmax - xmin);
    xtypdist = std::min(xtypdist, ymax - ymin);
  }

  return 1e-6 * xtypdist;
}

specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2>
assign_numbering(specfem::kokkos::HostView4d<double> global_coordinates,
                 const specfem::mesh::adjacency_map::adjacency_map<
                     specfem::dimension::type::dim2> &adjacency_map,
                 specfem::compute::mesh_to_compute_mapping &mapping) {

  int nspec = global_coordinates.extent(0);
  int ngll = global_coordinates.extent(1);
  int ngllxz = ngll * ngll;

  std::vector<qp> cart_cord(nspec * ngllxz);

  constexpr int chunk_size = specfem::parallel_config::storage_chunk_size;

  int iloc = 0;
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          cart_cord[iloc].x = global_coordinates(ispec, iz, ix, 0);
          cart_cord[iloc].z = global_coordinates(ispec, iz, ix, 1);
          cart_cord[iloc].iloc = iloc;
          iloc++;
        }
      }
    }
  }

  int nglob;
  if (adjacency_map.was_initialized()) {
    iloc = 0;
    specfem::kokkos::HostView3d<int> adjmap_index_mapping;
    std::tie(adjmap_index_mapping, nglob) =
        adjacency_map.generate_assembly_mapping(ngll);
    // const auto [adjmap_index_mapping, nglob_] =
    //     adjacency_map.generate_assembly_mapping(ngll);
    // nglob = nglob_;
    for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
      for (int iz = 0; iz < ngll; iz++) {
        for (int ix = 0; ix < ngll; ix++) {
          for (int ielement = 0; ielement < chunk_size; ielement++) {
            int ispec = ichunk + ielement;
            if (ispec >= nspec)
              break;
            const int ispec_mesh = mapping.compute_to_mesh(ispec);
            cart_cord[iloc].iglob = adjmap_index_mapping(ispec_mesh, iz, ix);
            iloc++;
          }
        }
      }
    }
  } else {
    // Sort cartesian coordinates in ascending order i.e.
    // cart_cord = [{0,0}, {0, 25}, {0, 50}, ..., {50, 0}, {50, 25}, {50, 50}]
    std::sort(cart_cord.begin(), cart_cord.end(),
              [&](const qp qp1, const qp qp2) {
                if (qp1.x != qp2.x) {
                  return qp1.x < qp2.x;
                }

                return qp1.z < qp2.z;
              });

    // Setup numbering
    int ig = 0;
    cart_cord[0].iglob = ig;

    type_real xtol = get_tolerance(cart_cord, nspec, ngllxz);

    for (int iloc = 1; iloc < cart_cord.size(); iloc++) {
      // check if the previous point is same as current
      if ((std::abs(cart_cord[iloc].x - cart_cord[iloc - 1].x) > xtol) ||
          (std::abs(cart_cord[iloc].z - cart_cord[iloc - 1].z) > xtol)) {
        ig++;
      }
      cart_cord[iloc].iglob = ig;
    }

    nglob = ig + 1;
  }

  std::vector<qp> copy_cart_cord(nspec * ngllxz);

  // reorder cart cord in original format
  for (int i = 0; i < cart_cord.size(); i++) {
    int iloc = cart_cord[i].iloc;
    copy_cart_cord[iloc] = cart_cord[i];
  }

  specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2> points(
      nspec, ngll, ngll);

  // Assign numbering to corresponding ispec, iz, ix
  std::vector<int> iglob_counted(nglob, -1);
  iloc = 0;
  int inum = 0;
  type_real xmin = std::numeric_limits<type_real>::max();
  type_real xmax = std::numeric_limits<type_real>::min();
  type_real zmin = std::numeric_limits<type_real>::max();
  type_real zmax = std::numeric_limits<type_real>::min();

  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          if (iglob_counted[copy_cart_cord[iloc].iglob] == -1) {

            const type_real x_cor = copy_cart_cord[iloc].x;
            const type_real z_cor = copy_cart_cord[iloc].z;
            if (xmin > x_cor)
              xmin = x_cor;
            if (zmin > z_cor)
              zmin = z_cor;
            if (xmax < x_cor)
              xmax = x_cor;
            if (zmax < z_cor)
              zmax = z_cor;

            iglob_counted[copy_cart_cord[iloc].iglob] = inum;
            points.h_index_mapping(ispec, iz, ix) = inum;
            points.h_coord(0, ispec, iz, ix) = x_cor;
            points.h_coord(1, ispec, iz, ix) = z_cor;
            inum++;
          } else {
            points.h_index_mapping(ispec, iz, ix) =
                iglob_counted[copy_cart_cord[iloc].iglob];
            points.h_coord(0, ispec, iz, ix) = copy_cart_cord[iloc].x;
            points.h_coord(1, ispec, iz, ix) = copy_cart_cord[iloc].z;
          }
          iloc++;
        }
      }
    }
  }

  points.xmin = xmin;
  points.xmax = xmax;
  points.zmin = zmin;
  points.zmax = zmax;

  assert(nglob != (nspec * ngllxz));

  assert(inum == nglob);

  Kokkos::deep_copy(points.index_mapping, points.h_index_mapping);
  Kokkos::deep_copy(points.coord, points.h_coord);

  return points;
}

} // namespace

specfem::assembly::mesh<specfem::dimension::type::dim2>::mesh(
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
        &control_nodes_in,
    const specfem::quadrature::quadratures &quadratures,
    const specfem::mesh::adjacency_map::adjacency_map<
        specfem::dimension::type::dim2> &adjacency_map) {
  nspec = tags.nspec;
  ngllz = quadratures.gll.get_N();
  ngllx = quadratures.gll.get_N();
  ngnod = control_nodes_in.ngnod;

  auto &mapping =
      static_cast<specfem::assembly::mesh_impl::mesh_to_compute_mapping<
          specfem::dimension::type::dim2> &>(*this);
  auto &control_nodes = static_cast<specfem::assembly::mesh_impl::control_nodes<
      specfem::dimension::type::dim2> &>(*this);
  auto &quadrature = static_cast<specfem::assembly::mesh_impl::quadrature<
      specfem::dimension::type::dim2> &>(*this);
  auto &shape_functions =
      static_cast<specfem::assembly::mesh_impl::shape_functions<
          specfem::dimension::type::dim2> &>(*this);

  mapping = specfem::assembly::mesh_impl::mesh_to_compute_mapping<
      specfem::dimension::type::dim2>(tags);
  control_nodes = specfem::assembly::mesh_impl::control_nodes<
      specfem::dimension::type::dim2>(
      *static_cast<const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
          specfem::dimension::type::dim2> *>(this),
      control_nodes_in);
  quadrature =
      specfem::assembly::mesh_impl::quadrature<specfem::dimension::type::dim2>(
          quadratures);

  shape_functions = specfem::assembly::mesh_impl::shape_functions<
      specfem::dimension::type::dim2>(
      quadratures.gll.get_hxi(), quadratures.gll.get_hxi(),
      quadratures.gll.get_N(), control_nodes_in.ngnod);

  this->assemble(adjacency_map);
}

void specfem::assembly::mesh<specfem::dimension::type::dim2>::assemble(
    const specfem::mesh::adjacency_map::adjacency_map<
        specfem::dimension::type::dim2> &adjacency_map) {

  const int ngnod = this->ngnod;
  const int nspec = this->nspec;

  const int ngll = this->ngllx; // = ngllz

  const auto xi = this->h_xi;
  const auto gamma = this->h_xi;

  const auto shape2D = this->h_shape2D;
  const auto coord = this->h_control_node_coord;

  const int scratch_size =
      specfem::kokkos::HostScratchView2d<type_real>::shmem_size(ndim, ngnod);

  specfem::kokkos::HostView4d<double> global_coordinates(
      "specfem::assembly::mesh::assemble::global_coordinates", nspec, ngll,
      ngll, 2);

  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        auto shape_functions =
            specfem::jacobian::define_shape_functions(xi(ix), gamma(iz), ngnod);

        double xcor = 0.0;
        double zcor = 0.0;

        for (int in = 0; in < ngnod; in++) {
          xcor += coord(0, ispec, in) * shape_functions[in];
          zcor += coord(1, ispec, in) * shape_functions[in];
        }

        global_coordinates(ispec, iz, ix, 0) = xcor;
        global_coordinates(ispec, iz, ix, 1) = zcor;
      }
    }
  }

  auto &points = static_cast<
      specfem::assembly::mesh_impl::points<specfem::dimension::type::dim2> &>(
      *this);

  points = assign_numbering(global_coordinates, adjacency_map, mapping);
}
