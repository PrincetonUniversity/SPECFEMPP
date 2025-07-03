#pragma once

#include "jacobian/interface.hpp"
#include "shape_functions.hpp"
#include <Kokkos_Core.hpp>

specfem::assembly::impl::shape_functions<specfem::dimension::type::dim2>::shape_functions(
    const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> xi,
    const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> gamma,
    const int &ngll, const int &ngnod)
    : ngllz(ngll), ngllx(ngll), ngnod(ngnod),
      shape2D("specfem::assembly::shape_functions::shape2D", ngll, ngll, ngnod),
      dshape2D("specfem::assembly::shape_functions::dshape2D", ngll, ngll, ndim,
               ngnod),
      h_shape2D(Kokkos::create_mirror_view(shape2D)),
      h_dshape2D(Kokkos::create_mirror_view(dshape2D)) {

  // Compute shape functions and their derivatives at quadrature points
  Kokkos::parallel_for(
      "shape_functions",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2> >({ 0, 0 }, { ngllz, ngllx }),
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
