#include "compute/interface.hpp"
#include "jacobian.h"
#include "kokkos_abstractions.h"
#include "shape_functions.h"
#include <Kokkos_Core.hpp>

specfem::compute::partial_derivatives::partial_derivatives(const int nspec,
                                                           const int ngllz,
                                                           const int ngllx)
    : xix(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::mesh::compute::xix", nspec, ngllz, ngllx)),
      xiz(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::mesh::compute::xiz", nspec, ngllz, ngllx)),
      gammax(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::mesh::compute::gammax", nspec, ngllz, ngllx)),
      gammaz(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::mesh::compute::gammaz", nspec, ngllz, ngllx)),
      jacobian(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::mesh::compute::jacobian", nspec, ngllz, ngllx)) {

  h_xix = Kokkos::create_mirror_view(xix);
  h_xiz = Kokkos::create_mirror_view(xiz);
  h_gammax = Kokkos::create_mirror_view(gammax);
  h_gammaz = Kokkos::create_mirror_view(gammaz);
  h_jacobian = Kokkos::create_mirror_view(jacobian);

  return;
};

specfem::compute::partial_derivatives::partial_derivatives(
    const specfem::kokkos::HostView2d<type_real> coorg,
    const specfem::kokkos::HostView2d<int> knods,
    const specfem::quadrature::quadrature &quadx,
    const specfem::quadrature::quadrature &quadz) {

  // Needs an axisymmetric update

  int ngnod = knods.extent(0);
  int nspec = knods.extent(1);

  int ngllx = quadx.get_N();
  int ngllz = quadz.get_N();
  int ngllxz = ngllx * ngllz;

  // Allocate views
  *this = specfem::compute::partial_derivatives(nspec, ngllz, ngllx);

  specfem::kokkos::HostMirror1d<type_real> xi = quadx.get_hxi();
  specfem::kokkos::HostMirror1d<type_real> gamma = quadz.get_hxi();

  // Allocate views for shape functions
  specfem::kokkos::HostView3d<type_real> shape2D(
      "specfem::mesh::assign_numbering::shape2D", ngllz, ngllx, ngnod);
  specfem::kokkos::HostView4d<type_real> dershape2D(
      "specfem::mesh::assign_numbering::dershape2D", ngllz, ngllx, ndim, ngnod);
  int scratch_size =
      specfem::kokkos::HostScratchView2d<type_real>::shmem_size(ndim, ngnod);

  // Allocate shape functions
  Kokkos::parallel_for(
      "shape_functions",
      specfem::kokkos::HostMDrange<2>({ 0, 0 }, { ngllz, ngllx }),
      [=](const int iz, const int ix) {
        type_real ixxi = xi(ix);
        type_real izgamma = gamma(iz);

        // Always use subviews inside parallel regions
        // ** Do not allocate views inside parallel regions **
        auto sv_shape2D = Kokkos::subview(shape2D, iz, ix, Kokkos::ALL);
        auto sv_dershape2D =
            Kokkos::subview(dershape2D, iz, ix, Kokkos::ALL, Kokkos::ALL);

        shape_functions::define_shape_functions(sv_shape2D, ixxi, izgamma,
                                                ngnod);
        shape_functions::define_shape_functions_derivatives(sv_dershape2D, ixxi,
                                                            izgamma, ngnod);
      });

  Kokkos::fence();

  Kokkos::parallel_for(
      specfem::kokkos::HostTeam(nspec, Kokkos::AUTO, ngnod)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      [=](const specfem::kokkos::HostTeam::member_type &teamMember) {
        const int ispec = teamMember.league_rank();

        //----- Load coorgx, coorgz in level 0 cache to be utilized later
        specfem::kokkos::HostScratchView2d<type_real> s_coorg(
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

              type_real jacobianl = jacobian::compute_jacobian(
                  teamMember, s_coorg, ngnod, sv_dershape2D);

              auto [xixl, gammaxl, xizl, gammazl] =
                  jacobian::compute_inverted_derivatives(teamMember, s_coorg,
                                                         ngnod, sv_dershape2D);

              this->h_xix(ispec, iz, ix) = xixl;
              this->h_gammax(ispec, iz, ix) = gammaxl;
              this->h_xiz(ispec, iz, ix) = xizl;
              this->h_gammaz(ispec, iz, ix) = gammazl;
              this->h_jacobian(ispec, iz, ix) = jacobianl;
            });
      });

  this->sync_views();

  return;
}

void specfem::compute::partial_derivatives::sync_views() {
  Kokkos::deep_copy(xix, h_xix);
  Kokkos::deep_copy(xiz, h_xiz);
  Kokkos::deep_copy(gammax, h_gammax);
  Kokkos::deep_copy(gammaz, h_gammaz);
  Kokkos::deep_copy(jacobian, h_jacobian);
}
