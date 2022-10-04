#include "../include/mesh.h"
#include "../include/boundaries.h"
#include "../include/compute.h"
#include "../include/config.h"
#include "../include/jacobian.h"
#include "../include/kokkos_abstractions.h"
#include "../include/material.h"
#include "../include/mpi_interfaces.h"
#include "../include/quadrature.h"
#include "../include/shape_functions.h"
#include "../include/specfem_mpi.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <limits>

specfem::materials::material_ind::material_ind(const int nspec,
                                               const int ngnod) {
  this->region_CPML =
      specfem::HostView1d<int>("specfem::mesh::region_CPML", nspec);
  this->kmato = specfem::HostView1d<int>("specfem::mesh::region_CPML", nspec);
  this->knods =
      specfem::HostView2d<int>("specfem::mesh::region_CPML", ngnod, nspec);

  for (int ispec = 0; ispec < nspec; ispec++) {
    this->kmato(ispec) = -1;
  }
  return;
}

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

specfem::HostView3d<int>
assign_numbering(const specfem::HostView2d<type_real> coorg,
                 const specfem::HostView2d<int> knods,
                 const quadrature::quadrature &quadx,
                 const quadrature::quadrature &quadz) {

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
      KOKKOS_LAMBDA(const int iz, const int ix) {
        type_real ixxi = xi(ix);
        type_real izgamma = gamma(iz);

        specfem::HostView1d<type_real> shape2D_tmp =
            shape_functions::define_shape_functions(ixxi, izgamma, ngnod);
        for (int in = 0; in < ngnod; in++)
          shape2D(iz, ix, in) = shape2D_tmp(in);
      });

  // Calculate the x and y coordinates for every GLL point

  Kokkos::parallel_for(
      specfem::HostTeam(nspec, Kokkos::AUTO, ngnod)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const specfem::HostTeam::member_type &teamMember) {
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

  // Assign numbering to corresponding ispec, iz, ix
  specfem::HostView3d<int> ibool("specfem::mesh::ibool", nspec, ngllz, ngllx);
  std::vector<int> iglob_counted(nglob, -1);
  int iloc = 0;
  int inum = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        if (iglob_counted[copy_cart_cord[iloc].iglob] == -1) {
          ibool(ispec, iz, ix) = inum;
          iglob_counted[copy_cart_cord[iloc].iglob] = inum;
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

  return ibool;
}

specfem::compute::coordinates
assign_mesh_coordinates(const specfem::HostView2d<type_real> coorg,
                        const specfem::HostView2d<int> knods,
                        const quadrature::quadrature &quadx,
                        const quadrature::quadrature &quadz) {

  // Needs an axisymmetric update

  int ngnod = knods.extent(0);
  int nspec = knods.extent(1);

  int ngllx = quadx.get_N();
  int ngllz = quadz.get_N();
  int ngllxz = ngllx * ngllz;

  specfem::HostMirror1d<type_real> xi = quadx.get_hxi();
  specfem::HostMirror1d<type_real> gamma = quadz.get_hxi();

  specfem::HostView3d<type_real> shape2D(
      "specfem::mesh::assign_numbering::shape2D", ngllz, ngllx, ngnod);
  specfem::HostView4d<type_real> dershape2D(
      "specfem::mesh::assign_numbering::dershape2D", ngllz, ngllx, ndim, ngnod);
  int scratch_size =
      specfem::HostScratchView2d<type_real>::shmem_size(ndim, ngnod);

  // Allocate shape functions
  Kokkos::parallel_for(
      "shape_functions", specfem::HostMDrange<2>({ 0, 0 }, { ngllz, ngllx }),
      KOKKOS_LAMBDA(const int iz, const int ix) {
        type_real ixxi = xi(ix);
        type_real izgamma = gamma(iz);

        specfem::HostView1d<type_real> shape2D_tmp =
            shape_functions::define_shape_functions(ixxi, izgamma, ngnod);
        specfem::HostView2d<type_real> dershape2D_tmp =
            shape_functions::define_shape_functions_derivatives(ixxi, izgamma,
                                                                ngnod);
        for (int in = 0; in < ngnod; in++)
          shape2D(iz, ix, in) = shape2D_tmp(in);
        for (int idim = 0; idim < ndim; idim++)
          for (int in = 0; in < ngnod; in++)
            dershape2D(iz, ix, idim, in) = dershape2D_tmp(idim, in);
      });

  specfem::compute::coordinates coordinates(nspec, ngllz, ngllx);

  Kokkos::parallel_for(
      specfem::HostTeam(nspec, Kokkos::AUTO, ngnod)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const specfem::HostTeam::member_type &teamMember) {
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

              // Get x and y coordinates for (ix, iz) point
              auto sv_shape2D = Kokkos::subview(shape2D, iz, ix, Kokkos::ALL);
              auto [xcor, ycor] = jacobian::compute_locations(
                  teamMember, s_coorg, ngnod, sv_shape2D);

              // compute partial derivatives
              auto sv_dershape2D =
                  Kokkos::subview(dershape2D, iz, ix, Kokkos::ALL, Kokkos::ALL);
              auto [xxi, zxi, xgamma, zgamma] =
                  jacobian::compute_partial_derivatives(teamMember, s_coorg,
                                                        ngnod, sv_dershape2D);

              type_real jacobianl =
                  jacobian::compute_jacobian(xxi, zxi, xgamma, zgamma);

              // invert the relation
              type_real xixl = zgamma / jacobianl;
              type_real gammaxl = -zxi / jacobianl;
              type_real xizl = -xgamma / jacobianl;
              type_real gammazl = xxi / jacobianl;

              coordinates.xcor(ispec, iz, ix) = xcor;
              coordinates.ycor(ispec, iz, ix) = ycor;
              coordinates.xix(ispec, iz, ix) = xixl;
              coordinates.gammax(ispec, iz, ix) = gammaxl;
              coordinates.xiz(ispec, iz, ix) = xizl;
              coordinates.gammaz(ispec, iz, ix) = gammazl;
              coordinates.jacobian(ispec, iz, ix) = jacobianl;
            });
      });

  return coordinates;
}

// void setup_mesh_periodic_edges(const int numacforcing){
//     if (numacforcing > 0){
//         throw std::runtime_error("Periodic edges are not implemented yet");
//     }
// }

void setup_mesh_acforcing_edges(const int numacforcing) {
  if (numacforcing > 0) {
    throw std::runtime_error("acoustic forcing edges are not implemented yet");
  }
}

specfem::compute::properties
setup_mesh_properties(specfem::HostView1d<int> kmato,
                      std::vector<specfem::material *> &materials,
                      const int nspec, const int ngllz, const int ngllx) {
  // Setup mesh properties
  // UPDATEME::
  //           acoustic materials
  //           poroelastic materials
  //           axisymmetric materials
  //           anisotropic materials

  specfem::compute::properties properties(nspec, ngllz, ngllx);

  Kokkos::parallel_for(
      "setup_mesh_properties",
      specfem::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int ispec, const int iz, const int ix) {
        const int imat = kmato(ispec);
        utilities::return_holder holder = materials[imat]->get_properties();
        auto [rho, mu, kappa, qmu, qkappa] = std::make_tuple(
            holder.rho, holder.mu, holder.kappa, holder.qmu, holder.qkappa);
        properties.rho(ispec, iz, ix) = rho;
        properties.mu(ispec, iz, ix) = mu;
        properties.kappa(ispec, iz, ix) = kappa;

        properties.qmu(ispec, iz, ix) = qmu;
        properties.qkappa(ispec, iz, ix) = qkappa;

        type_real vp = std::sqrt((kappa + mu) / rho);
        type_real vs = std::sqrt(mu / rho);

        properties.rho_vp(ispec, iz, ix) = rho * vp;
        properties.rho_vs(ispec, iz, ix) = rho * vs;
      });

  return properties;
}

void specfem::mesh::setup(std::vector<specfem::material *> &materials,
                          const quadrature::quadrature &quadx,
                          const quadrature::quadrature &quadz,
                          specfem::MPI *mpi) {

  this->compute.ibool =
      assign_numbering(this->coorg, this->material_ind.knods, quadx, quadz);

  mpi->sync_all();

  this->compute.coordinates = assign_mesh_coordinates(
      this->coorg, this->material_ind.knods, quadx, quadz);

  mpi->sync_all();

  setup_mesh_acforcing_edges(parameters.nelem_acforcing);

  mpi->sync_all();

  int ngllx = quadx.get_N();
  int ngllz = quadz.get_N();

  this->compute.properties = setup_mesh_properties(
      this->material_ind.kmato, materials, this->nspec, ngllx, ngllz);

  mpi->sync_all();
}

// specfem::mesh::mesh(std::filename){
//   auto [mesh_tmp, materials] =
// }
