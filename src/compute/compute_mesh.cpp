#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "jacobian/interface.hpp"
#include "kokkos_abstractions.h"
#include "parallel_configuration/chunk_config.hpp"
#include "quadrature/interface.hpp"
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

specfem::compute::points
assign_numbering(specfem::kokkos::HostView4d<double> global_coordinates) {

  int nspec = global_coordinates.extent(0);
  int ngll = global_coordinates.extent(1);
  int ngllxz = ngll * ngll;

  std::vector<qp> cart_cord(nspec * ngllxz);

  using simd = specfem::datatype::simd<type_real, true>;

  constexpr int chunk_size = specfem::parallel_config::storage_chunk_size;

  int nchunks = nspec / chunk_size;
  int iloc = 0;
  for (int ichunk = 0; ichunk < nchunks; ichunk++) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk * chunk_size + ielement;
          cart_cord[iloc].x = global_coordinates(ispec, iz, ix, 0);
          cart_cord[iloc].z = global_coordinates(ispec, iz, ix, 1);
          cart_cord[iloc].iloc = iloc;
          iloc++;
        }
      }
    }
  }

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

  std::vector<qp> copy_cart_cord(nspec * ngllxz);

  // reorder cart cord in original format
  for (int i = 0; i < cart_cord.size(); i++) {
    int iloc = cart_cord[i].iloc;
    copy_cart_cord[iloc] = cart_cord[i];
  }

  int nglob = ig + 1;

  specfem::compute::points points(nspec, ngll, ngll);

  // Assign numbering to corresponding ispec, iz, ix
  std::vector<int> iglob_counted(nglob, -1);
  iloc = 0;
  int inum = 0;
  type_real xmin = std::numeric_limits<type_real>::max();
  type_real xmax = std::numeric_limits<type_real>::min();
  type_real zmin = std::numeric_limits<type_real>::max();
  type_real zmax = std::numeric_limits<type_real>::min();

  for (int ichunk = 0; ichunk < nchunks; ichunk++) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk * chunk_size + ielement;
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

specfem::compute::control_nodes::control_nodes(
    const specfem::compute::mesh_to_compute_mapping &mapping,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
        &control_nodes)
    : ngnod(control_nodes.ngnod), nspec(control_nodes.nspec),
      index_mapping("specfem::compute::control_nodes::index_mapping",
                    control_nodes.nspec, control_nodes.ngnod),
      coord("specfem::compute::control_nodes::coord", ndim, control_nodes.nspec,
            control_nodes.ngnod),
      h_index_mapping(Kokkos::create_mirror_view(index_mapping)),
      h_coord(Kokkos::create_mirror_view(coord)) {

  Kokkos::parallel_for(
      "specfem::compute::control_nodes::assign_index_mapping",
      specfem::kokkos::HostMDrange<2>({ 0, 0 }, { ngnod, nspec }),
      [=](const int in, const int ispec) {
        const int ispec_mesh = mapping.compute_to_mesh(ispec);
        const int index = control_nodes.knods(in, ispec_mesh);
        h_index_mapping(ispec, in) = index;
        h_coord(0, ispec, in) = control_nodes.coord(0, index);
        h_coord(1, ispec, in) = control_nodes.coord(1, index);
      });

  Kokkos::deep_copy(index_mapping, h_index_mapping);
  Kokkos::deep_copy(coord, h_coord);

  return;
}

specfem::compute::shape_functions::shape_functions(
    const specfem::kokkos::HostMirror1d<type_real> xi,
    const specfem::kokkos::HostMirror1d<type_real> gamma, const int &ngll,
    const int &ngnod)
    : ngllz(ngll), ngllx(ngll), ngnod(ngnod),
      shape2D("specfem::compute::shape_functions::shape2D", ngll, ngll, ngnod),
      dshape2D("specfem::compute::shape_functions::dshape2D", ngll, ngll, ndim,
               ngnod),
      h_shape2D(Kokkos::create_mirror_view(shape2D)),
      h_dshape2D(Kokkos::create_mirror_view(dshape2D)) {

  // Compute shape functions and their derivatives at quadrature points
  Kokkos::parallel_for(
      "shape_functions",
      specfem::kokkos::HostMDrange<2>({ 0, 0 }, { ngll, ngll }),
      [=](const int iz, const int ix) {
        type_real xil = xi(ix);
        type_real gammal = gamma(iz);

        // Always use subviews inside parallel regions
        // ** Do not allocate views inside parallel regions **
        auto sv_shape2D = Kokkos::subview(h_shape2D, iz, ix, Kokkos::ALL);
        auto sv_dshape2D =
            Kokkos::subview(h_dshape2D, iz, ix, Kokkos::ALL, Kokkos::ALL);
        specfem::jacobian::define_shape_functions(sv_shape2D, xil, gammal,
                                                  ngnod);

        specfem::jacobian::define_shape_functions_derivatives(sv_dshape2D, xil,
                                                              gammal, ngnod);
      });

  Kokkos::deep_copy(shape2D, h_shape2D);
  Kokkos::deep_copy(dshape2D, h_dshape2D);

  return;
}

specfem::compute::mesh_to_compute_mapping::mesh_to_compute_mapping(
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags)
    : compute_to_mesh("specfem::compute::mesh_to_compute_mapping", tags.nspec),
      mesh_to_compute("specfem::compute::mesh_to_compute_mapping", tags.nspec) {

  const int nspec = tags.nspec;

  constexpr auto element_types = specfem::element::element_types();
  constexpr int total_element_types = element_types.size();

  std::array<std::vector<int>, total_element_types> element_type_ispec;
  int total_counted = 0;

  for (int i = 0; i < total_element_types; i++) {
    const auto [dimension, medium_tag, property_tag, boundary_tag] =
        element_types[i];
    for (int ispec = 0; ispec < nspec; ispec++) {
      const auto tag = tags.tags_container(ispec);
      if (tag.medium_tag == medium_tag && tag.property_tag == property_tag &&
          tag.boundary_tag == boundary_tag) {
        element_type_ispec[i].push_back(ispec);
      }
    }
    total_counted += element_type_ispec[i].size();
  }

  assert(total_counted == nspec);

  int ispec = 0;

  for (const auto &element_ispec : element_type_ispec) {
    for (const auto &ispecs : element_ispec) {
      compute_to_mesh(ispec) = ispecs;
      mesh_to_compute(ispecs) = ispec;
      ispec++;
    }
  }

  assert(ispec == nspec);
}

specfem::compute::mesh::mesh(
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
        &m_control_nodes,
    const specfem::quadrature::quadratures &m_quadratures) {

  this->mapping = specfem::compute::mesh_to_compute_mapping(tags);
  this->control_nodes =
      specfem::compute::control_nodes(this->mapping, m_control_nodes);
  this->quadratures =
      specfem::compute::quadrature(m_quadratures, m_control_nodes);
  this->nspec = this->control_nodes.nspec;
  this->ngllx = this->quadratures.gll.N;
  this->ngllz = this->quadratures.gll.N;

  this->points = this->assemble();
}

specfem::compute::points specfem::compute::mesh::assemble() {

  const int ngnod = control_nodes.ngnod;
  const int nspec = control_nodes.nspec;

  const int ngll = quadratures.gll.N;
  const int ngllxz = ngll * ngll;

  const auto xi = quadratures.gll.h_xi;
  const auto gamma = quadratures.gll.h_xi;

  const auto shape2D = this->quadratures.gll.shape_functions.h_shape2D;
  const auto coord = this->control_nodes.h_coord;

  const int scratch_size =
      specfem::kokkos::HostScratchView2d<type_real>::shmem_size(ndim, ngnod);

  specfem::kokkos::HostView4d<double> global_coordinates(
      "specfem::compute::mesh::assemble::global_coordinates", nspec, ngll, ngll,
      2);

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

  // // Compute the cartesian coordinates of the GLL points
  // Kokkos::parallel_for(
  //     specfem::kokkos::HostTeam(nspec, Kokkos::AUTO, Kokkos::AUTO)
  //         .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
  //     KOKKOS_CLASS_LAMBDA(
  //         const specfem::kokkos::HostTeam::member_type teamMember) {
  //       const int ispec = teamMember.league_rank();

  //       //----- Load coorgx, coorgz in level 0 cache to be utilized later
  //       specfem::kokkos::HostScratchView2d<type_real> s_coord(
  //           teamMember.team_scratch(0), ndim, ngnod);

  //       Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, ngnod),
  //                            [&](const int in) {
  //                              s_coord(0, in) = coord(0, ispec, in);
  //                              s_coord(1, in) = coord(1, ispec, in);
  //                            });

  //       teamMember.team_barrier();
  //       //-----

  //       Kokkos::parallel_for(
  //           Kokkos::TeamThreadRange(teamMember, ngllxz), [=](const int xz) {
  //             int ix, iz;
  //             sub2ind(xz, ngll, iz, ix);
  //             // Get x and y coordinates for (ix, iz) point
  //             auto sv_shape2D = Kokkos::subview(shape2D, iz, ix,
  //             Kokkos::ALL); auto [xcor, zcor] = jacobian::compute_locations(
  //                 teamMember, s_coord, ngnod, sv_shape2D);
  //             // ------------

  //             global_coordinates(ispec, iz, ix) = { xcor, zcor };
  //           });
  //     });

  // Kokkos::fence();

  return assign_numbering(global_coordinates);
}

// specfem::compute::compute::compute(
//     const specfem::kokkos::HostView2d<type_real> coorg,
//     const specfem::kokkos::HostView2d<int> knods,
//     const specfem::quadrature::quadrature *quadx,
//     const specfem::quadrature::quadrature *quadz) {

//   int ngnod = knods.extent(0);
//   int nspec = knods.extent(1);

//   int ngllx = quadx->get_N();
//   int ngllz = quadz->get_N();
//   int ngllxz = ngllx * ngllz;

//   *this = specfem::compute::compute(nspec, ngllz, ngllx);

//   specfem::kokkos::HostMirror1d<type_real> xi = quadx->get_hxi();
//   specfem::kokkos::HostMirror1d<type_real> gamma = quadz->get_hxi();
//   specfem::kokkos::HostView3d<type_real> shape2D(
//       "specfem::mesh::assign_numbering", ngllz, ngllx, ngnod);

//   std::vector<qp> cart_cord(nspec * ngllxz);
//   specfem::kokkos::HostView1d<qp> pcart_cord(
//       "specfem::compute::compute::pcart_cord", nspec * ngllxz);
//   int scratch_size =
//       specfem::kokkos::HostScratchView2d<type_real>::shmem_size(ndim,
//       ngnod);

//   // Allocate shape functions
//   Kokkos::parallel_for(
//       "shape_functions",
//       specfem::kokkos::HostMDrange<2>({ 0, 0 }, { ngllz, ngllx }),
//       [=](const int iz, const int ix) {
//         type_real ixxi = xi(ix);
//         type_real izgamma = gamma(iz);

//         // Always use subviews inside parallel regions
//         // ** Do not allocate views inside parallel regions **
//         auto sv_shape2D = Kokkos::subview(shape2D, iz, ix, Kokkos::ALL);
//         specfem::jacobian::define_shape_functions(sv_shape2D, ixxi,
//         izgamma,
//                                                   ngnod);
//       });

//   // Calculate the x and y coordinates for every GLL point

//   Kokkos::parallel_for(
//       specfem::kokkos::HostTeam(nspec, Kokkos::AUTO, ngnod)
//           .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
//       [=](const specfem::kokkos::HostTeam::member_type teamMember) {
//         const int ispec = teamMember.league_rank();

//         //----- Load coorgx, coorgz in level 0 cache to be utilized later
//         specfem::kokkos::HostScratchView2d<type_real> s_coorg(
//             teamMember.team_scratch(0), ndim, ngnod);

//         // This loop is not vectorizable because access to coorg via
//         // knods(ispec, in) is not vectorizable
//         Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, ngnod),
//                              [&](const int in) {
//                                s_coorg(0, in) = coorg(0, knods(in, ispec));
//                                s_coorg(1, in) = coorg(1, knods(in, ispec));
//                              });

//         teamMember.team_barrier();
//         //-----

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(teamMember, ngllxz), [&](const int xz)
//             {
//               int ix, iz;
//               sub2ind(xz, ngllx, iz, ix);
//               const int iloc = ispec * (ngllxz) + xz;

//               // Get x and y coordinates for (ix, iz) point
//               auto sv_shape2D = Kokkos::subview(shape2D, iz, ix,
//               Kokkos::ALL); auto [xcor, ycor] =
//               jacobian::compute_locations(
//                   teamMember, s_coorg, ngnod, sv_shape2D);
//               // ------------
//               // Hacky way of doing this (but nacessary), because
//               // KOKKOS_LAMBDA is a const operation so I cannot update
//               // cart_cord inside of a lambda directly Since iloc is
//               // different within every thread I ensure that I don't have a
//               // race condition here.
//               pcart_cord(iloc).x = xcor;
//               pcart_cord(iloc).y = ycor;
//               pcart_cord(iloc).iloc = iloc;
//             });
//       });

//   for (int iloc = 0; iloc < nspec * ngllxz; iloc++) {
//     cart_cord[iloc] = pcart_cord(iloc);
//   }

//   std::tie(this->coordinates.coord, this->coordinates.xmin,
//            this->coordinates.xmax, this->coordinates.zmin,
//            this->coordinates.zmax) =
//       assign_numbering(this->h_ibool, cart_cord, nspec, ngllx, ngllz);

//   this->sync_views();
// }

// void specfem::compute::compute::sync_views() {
//   Kokkos::deep_copy(ibool, h_ibool);

//   return;
// }
