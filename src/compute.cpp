#include "../include/compute.h"
#include "../include/jacobian.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/shape_functions.h"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

struct qp {
  type_real x = 0, y = 0;
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
      ymax = std::max(ymax, cart_cord[iloc].y);
      ymin = std::min(ymin, cart_cord[iloc].y);
    }

    xtypdist = std::min(xtypdist, xmax - xmin);
    xtypdist = std::min(xtypdist, ymax - ymin);
  }

  return 1e-6 * xtypdist;
}

std::tuple<specfem::HostView3d<int>, specfem::HostView2d<type_real>, type_real,
           type_real, type_real, type_real>
assign_numbering(std::vector<qp> &cart_cord, const int nspec, const int ngllx,
                 const int ngllz) {

  int ngllxz = ngllx * ngllz;
  // Sort cartesian coordinates in ascending order i.e.
  // cart_cord = [{0,0}, {0, 25}, {0, 50}, ..., {50, 0}, {50, 25}, {50, 50}]
  std::sort(cart_cord.begin(), cart_cord.end(),
            [&](const qp qp1, const qp qp2) {
              if (qp1.x != qp2.x) {
                return qp1.x < qp2.x;
              }

              return qp1.y < qp2.y;
            });

  // Setup numbering
  int ig = 0;
  cart_cord[0].iglob = ig;

  type_real xtol = get_tolerance(cart_cord, nspec, ngllxz);

  for (int iloc = 1; iloc < cart_cord.size(); iloc++) {
    // check if the previous point is same as current
    if ((std::abs(cart_cord[iloc].x - cart_cord[iloc - 1].x) > xtol) ||
        (std::abs(cart_cord[iloc].y - cart_cord[iloc - 1].y) > xtol)) {
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

  specfem::HostView3d<int> ibool("specfem::mesh::ibool", nspec, ngllz, ngllx);
  specfem::HostView2d<type_real> coord("specfem::mesh::coord", ndim, nglob);
  // Assign numbering to corresponding ispec, iz, ix
  std::vector<int> iglob_counted(nglob, -1);
  int iloc = 0;
  int inum = 0;
  type_real xmin = std::numeric_limits<type_real>::max();
  type_real xmax = std::numeric_limits<type_real>::min();
  type_real zmin = std::numeric_limits<type_real>::max();
  type_real zmax = std::numeric_limits<type_real>::min();
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        if (iglob_counted[copy_cart_cord[iloc].iglob] == -1) {
          ibool(ispec, iz, ix) = inum;
          iglob_counted[copy_cart_cord[iloc].iglob] = inum;
          coord(0, inum) = copy_cart_cord[iloc].x;
          coord(1, inum) = copy_cart_cord[iloc].y;
          if (xmin > coord(0, inum))
            xmin = coord(0, inum);
          if (zmin > coord(1, inum))
            zmin = coord(1, inum);
          if (xmax < coord(0, inum))
            xmax = coord(0, inum);
          if (zmax < coord(1, inum))
            zmax = coord(0, inum);
          inum++;
        } else {
          ibool(ispec, iz, ix) = iglob_counted[copy_cart_cord[iloc].iglob];
        }
        iloc++;
      }
    }
  }

  assert(nglob != (nspec * ngllxz));

  assert(inum == nglob);

  return std::make_tuple(ibool, coord, xmin, xmax, zmin, zmax);
}

specfem::compute::compute::compute(
    const specfem::HostView2d<type_real> coorg,
    const specfem::HostView2d<int> knods,
    const specfem::quadrature::quadrature &quadx,
    const specfem::quadrature::quadrature &quadz) {

  int ngnod = knods.extent(0);
  int nspec = knods.extent(1);

  int ngllx = quadx.get_N();
  int ngllz = quadz.get_N();
  int ngllxz = ngllx * ngllz;

  specfem::HostMirror1d<type_real> xi = quadx.get_hxi();
  specfem::HostMirror1d<type_real> gamma = quadz.get_hxi();
  specfem::HostView3d<type_real> shape2D("specfem::mesh::assign_numbering",
                                         ngllz, ngllx, ngnod);

  std::vector<qp> cart_cord(nspec * ngllxz);
  std::vector<qp> *pcart_cord = &cart_cord;
  int scratch_size =
      specfem::HostScratchView2d<type_real>::shmem_size(ndim, ngnod);

  // Allocate shape functions
  Kokkos::parallel_for(
      "shape_functions", specfem::HostMDrange<2>({ 0, 0 }, { ngllz, ngllx }),
      [=](const int iz, const int ix) {
        type_real ixxi = xi(ix);
        type_real izgamma = gamma(iz);

        // Always use subviews inside parallel regions
        // ** Do not allocate views inside parallel regions **
        auto sv_shape2D = Kokkos::subview(shape2D, iz, ix, Kokkos::ALL);
        shape_functions::define_shape_functions(sv_shape2D, ixxi, izgamma,
                                                ngnod);
      });

  // Calculate the x and y coordinates for every GLL point

  Kokkos::parallel_for(
      specfem::HostTeam(nspec, Kokkos::AUTO, ngnod)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      [=](const specfem::HostTeam::member_type teamMember) {
        const int ispec = teamMember.league_rank();

        //----- Load coorgx, coorgz in level 0 cache to be utilized later
        specfem::HostScratchView2d<type_real> s_coorg(
            teamMember.team_scratch(0), ndim, ngnod);

        // This loop is not vectorizable because access to coorg via
        // knods(ispec, in) is not vectorizable
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, ngnod),
                             [&](const int in) {
                               s_coorg(0, in) = coorg(0, knods(in, ispec));
                               s_coorg(1, in) = coorg(1, knods(in, ispec));
                             });

        teamMember.team_barrier();
        //-----

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, ngllxz), [&](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;
              const int iloc = ispec * (ngllxz) + xz;

              // Get x and y coordinates for (ix, iz) point
              auto sv_shape2D = Kokkos::subview(shape2D, iz, ix, Kokkos::ALL);
              auto [xcor, ycor] = jacobian::compute_locations(
                  teamMember, s_coorg, ngnod, sv_shape2D);
              // ------------
              // Hacky way of doing this (but nacessary), because
              // KOKKOS_LAMBDA is a const operation so I cannot update
              // cart_cord inside of a lambda directly Since iloc is
              // different within every thread I ensure that I don't have a
              // race condition here.
              (*pcart_cord)[iloc].x = xcor;
              (*pcart_cord)[iloc].y = ycor;
              (*pcart_cord)[iloc].iloc = iloc;
            });
      });

  std::tie(this->ibool, this->coordinates.coord, this->coordinates.xmin,
           this->coordinates.xmax, this->coordinates.zmin,
           this->coordinates.zmax) =
      assign_numbering(cart_cord, nspec, ngllx, ngllz);
}
