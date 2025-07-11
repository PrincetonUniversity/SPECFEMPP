#pragma once

#include "jacobian/interface.hpp"
#include "shape_functions.hpp"
#include "specfem/shape_functions.hpp"
#include <Kokkos_Core.hpp>

specfem::assembly::mesh_impl::shape_functions<specfem::dimension::type::dim2>::
    shape_functions(
        const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> xi,
        const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>
            gamma,
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
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<2> >({ 0, 0 }, { ngllz, ngllx }),
      [=](const int iz, const int ix) {
        type_real xil = xi(ix);
        type_real gammal = gamma(iz);

        const auto shape_function =
            specfem::shape_function::shape_function(xil, gammal, ngnod);

        for (int in = 0; in < ngnod; in++) {
          h_shape2D(iz, ix, in) = shape_function[in];
        }

        const auto shape_function_derivatives =
            specfem::shape_function::shape_function_derivatives(xil, gammal,
                                                                ngnod);

        for (int in = 0; in < ngnod; in++) {
          h_dshape2D(iz, ix, 0, in) = shape_function_derivatives[0][in];
          h_dshape2D(iz, ix, 1, in) = shape_function_derivatives[1][in];
        }
      });

  Kokkos::deep_copy(shape2D, h_shape2D);
  Kokkos::deep_copy(dshape2D, h_dshape2D);

  return;
}
