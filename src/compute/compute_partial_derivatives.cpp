#include "compute/interface.hpp"
#include "jacobian/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

specfem::compute::partial_derivatives::partial_derivatives(const int nspec,
                                                           const int ngllz,
                                                           const int ngllx)
    : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
      xix("specfem::compute::partial_derivatives::xix", nspec, ngllz, ngllx),
      xiz("specfem::compute::partial_derivatives::xiz", nspec, ngllz, ngllx),
      gammax("specfem::compute::partial_derivatives::gammax", nspec, ngllz,
             ngllx),
      gammaz("specfem::compute::partial_derivatives::gammaz", nspec, ngllz,
             ngllx),
      jacobian("specfem::compute::partial_derivatives::jacobian", nspec, ngllz,
               ngllx),
      h_xix(specfem::kokkos::create_mirror_view(xix)),
      h_xiz(specfem::kokkos::create_mirror_view(xiz)),
      h_gammax(specfem::kokkos::create_mirror_view(gammax)),
      h_gammaz(specfem::kokkos::create_mirror_view(gammaz)),
      h_jacobian(specfem::kokkos::create_mirror_view(jacobian)) {
  return;
};

specfem::compute::partial_derivatives::partial_derivatives(
    const specfem::compute::mesh &mesh)
    : nspec(mesh.control_nodes.nspec), ngllz(mesh.quadratures.gll.N),
      ngllx(mesh.quadratures.gll.N),
      xix("specfem::compute::partial_derivatives::xix", nspec, ngllz, ngllx),
      xiz("specfem::compute::partial_derivatives::xiz", nspec, ngllz, ngllx),
      gammax("specfem::compute::partial_derivatives::gammax", nspec, ngllz,
             ngllx),
      gammaz("specfem::compute::partial_derivatives::gammaz", nspec, ngllz,
             ngllx),
      jacobian("specfem::compute::partial_derivatives::jacobian", nspec, ngllz,
               ngllx),
      h_xix(specfem::kokkos::create_mirror_view(xix)),
      h_xiz(specfem::kokkos::create_mirror_view(xiz)),
      h_gammax(specfem::kokkos::create_mirror_view(gammax)),
      h_gammaz(specfem::kokkos::create_mirror_view(gammaz)),
      h_jacobian(specfem::kokkos::create_mirror_view(jacobian)) {

  const int ngnod = mesh.control_nodes.ngnod;
  const int ngllxz = ngllz * ngllx;

  const int scratch_size =
      specfem::kokkos::HostScratchView2d<type_real>::shmem_size(ndim, ngnod);

  Kokkos::parallel_for(
      specfem::kokkos::HostTeam(nspec, Kokkos::AUTO, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      [=](const specfem::kokkos::HostTeam::member_type &teamMember) {
        const int ispec = teamMember.league_rank();

        //----- Load coorgx, coorgz in level 0 cache to be utilized later
        specfem::kokkos::HostScratchView2d<type_real> s_coorg(
            teamMember.team_scratch(0), ndim, ngnod);

        // This loop is not vectorizable because access to coorg via
        // knods(ispec, in) is not vectorizable
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, ngnod), [&](const int in) {
              s_coorg(0, in) = mesh.control_nodes.h_coord(0, ispec, in);
              s_coorg(1, in) = mesh.control_nodes.h_coord(1, ispec, in);
            });

        teamMember.team_barrier();
        //-----

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, ngllxz), [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              // compute partial derivatives
              auto sv_dershape2D = Kokkos::subview(
                  mesh.quadratures.gll.shape_functions.h_dshape2D, iz, ix,
                  Kokkos::ALL, Kokkos::ALL);

              auto derivatives = jacobian::compute_derivatives(
                  teamMember, s_coorg, ngnod, sv_dershape2D);

              this->h_xix(ispec, iz, ix) = derivatives.xix;
              this->h_gammax(ispec, iz, ix) = derivatives.gammax;
              this->h_xiz(ispec, iz, ix) = derivatives.xiz;
              this->h_gammaz(ispec, iz, ix) = derivatives.gammaz;
              this->h_jacobian(ispec, iz, ix) = derivatives.jacobian;
            });
      });

  specfem::kokkos::deep_copy(xix, h_xix);
  specfem::kokkos::deep_copy(xiz, h_xiz);
  specfem::kokkos::deep_copy(gammax, h_gammax);
  specfem::kokkos::deep_copy(gammaz, h_gammaz);
  specfem::kokkos::deep_copy(jacobian, h_jacobian);

  return;
}

void specfem::compute::partial_derivatives::sync_views() {
  specfem::kokkos::deep_copy(xix, h_xix);
  specfem::kokkos::deep_copy(xiz, h_xiz);
  specfem::kokkos::deep_copy(gammax, h_gammax);
  specfem::kokkos::deep_copy(gammaz, h_gammaz);
  specfem::kokkos::deep_copy(jacobian, h_jacobian);
}
