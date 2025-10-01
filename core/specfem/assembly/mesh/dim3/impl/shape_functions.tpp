#pragma once

#include "shape_functions.hpp"
#include "specfem/shape_functions.hpp"
#include <Kokkos_Core.hpp>

specfem::assembly::mesh_impl::shape_functions<specfem::dimension::type::dim3>::
    shape_functions(
        const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> xi,
        const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> eta,
        const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> zeta,
        const int &ngll, const int &ngnod)
    : ngllz(ngll), nglly(ngll), ngllx(ngll), ngnod(ngnod),
      shape3D("specfem::assembly::shape_functions::shape3D", ngll, ngll, ngll,
              ngnod),
      dshape3D("specfem::assembly::shape_functions::dshape3D", ngll, ngll, ngll,
               ndim, ngnod),
      h_shape3D(Kokkos::create_mirror_view(shape3D)),
      h_dshape3D(Kokkos::create_mirror_view(dshape3D)) {

  // Compute shape functions and their derivatives at quadrature points
  Kokkos::parallel_for(
      "shape_functions",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<3> >({ 0, 0, 0 },
                                              { ngllz, nglly, ngllx }),
      [=](const int iz, const int iy, const int ix) {
        type_real xil = xi(ix);
        type_real etal = eta(iy);
        type_real zetal = zeta(iz);

        const auto shape_function =
            specfem::shape_function::shape_function(xil, etal, zetal, ngnod);

        for (int in = 0; in < ngnod; in++) {
          h_shape3D(iz, iy, ix, in) = shape_function[in];
        }

        const auto shape_function_derivatives =
            specfem::shape_function::shape_function_derivatives(xil, etal,
                                                                zetal, ngnod);

        for (int in = 0; in < ngnod; in++) {
          h_dshape3D(iz, iy, ix, 0, in) = shape_function_derivatives[0][in];
          h_dshape3D(iz, iy, ix, 1, in) = shape_function_derivatives[1][in];
          h_dshape3D(iz, iy, ix, 2, in) = shape_function_derivatives[2][in];
        }
      });

  Kokkos::deep_copy(shape3D, h_shape3D);
  Kokkos::deep_copy(dshape3D, h_dshape3D);

  return;
}
