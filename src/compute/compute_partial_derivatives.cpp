#include "compute/interface.hpp"
#include "jacobian/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include <Kokkos_Core.hpp>

specfem::compute::partial_derivatives::partial_derivatives(const int nspec,
                                                           const int ngllz,
                                                           const int ngllx)
    : xix("specfem::compute::partial_derivatives::xix", nspec, ngllz, ngllx),
      xiz("specfem::compute::partial_derivatives::xiz", nspec, ngllz, ngllx),
      gammax("specfem::compute::partial_derivatives::gammax", nspec, ngllz,
             ngllx),
      gammaz("specfem::compute::partial_derivatives::gammaz", nspec, ngllz,
             ngllx),
      jacobian("specfem::compute::partial_derivatives::jacobian", nspec, ngllz,
               ngllx),
      h_xix(Kokkos::create_mirror_view(xix)),
      h_xiz(Kokkos::create_mirror_view(xiz)),
      h_gammax(Kokkos::create_mirror_view(gammax)),
      h_gammaz(Kokkos::create_mirror_view(gammaz)),
      h_jacobian(Kokkos::create_mirror_view(jacobian)) {
  return;
};

specfem::compute::partial_derivatives::partial_derivatives(
    const specfem::compute::mesh &mesh,
    const specfem::compute::quadrature &quadrature) {

  // Needs an axisymmetric update

  // I have to port this section to GPU

  const int nspec = mesh.nspec;
  const int ngll = quadrature.ngll;

  *this = specfem::compute::partial_derivatives(nspec, ngll, ngll);

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
              s_coorg(1, in) = mesh.control_nodes.h_coord(1, ispec, in)
            });

        teamMember.team_barrier();
        //-----

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, ngllxz), [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              // compute partial derivatives
              auto sv_dershape2D =
                  Kokkos::subview(mesh.shape_functions.h_dshape2D, iz, ix,
                                  Kokkos::ALL, Kokkos::ALL);

              auto elemental_derivatives =
                  jacobian::compute_inverted_derivatives(teamMember, s_coorg,
                                                         ngnod, sv_dershape2D);

              this->h_xix(ispec, iz, ix) = elemental_derivatives.xix;
              this->h_gammax(ispec, iz, ix) = elemental_derivatives.gammax;
              this->h_xiz(ispec, iz, ix) = elemental_derivatives.xiz;
              this->h_gammaz(ispec, iz, ix) = elemental_derivatives.gammaz;
              this->h_jacobian(ispec, iz, ix) = elemental_derivatives.jacobian;
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

// KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, 2>
// specfem::compute::element_partial_derivatives::compute_normal(
//     const specfem::enums::boundaries::type type) const {

//   switch (type) {
//   case specfem::enums::boundaries::type::BOTTOM:
//     return this->compute_normal<specfem::enums::boundaries::type::BOTTOM>();
//     break;
//   case specfem::enums::boundaries::type::TOP:
//     return this->compute_normal<specfem::enums::boundaries::type::TOP>();
//     break;
//   case specfem::enums::boundaries::type::LEFT:
//     return this->compute_normal<specfem::enums::boundaries::type::LEFT>();
//     break;
//   case specfem::enums::boundaries::type::RIGHT:
//     return this->compute_normal<specfem::enums::boundaries::type::RIGHT>();
//     break;
//   default:
// #ifndef NDEBUG
//     ASSERT(false, "Invalid boundary type");
// #endif
//     break;
//   }

//   return specfem::kokkos::array_type<type_real, 2>();
// }
