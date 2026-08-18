#pragma once

#include "shape_functions.hpp"
#include "specfem/shape_function.hpp"
#include <Kokkos_Core.hpp>

specfem::assembly::mesh_impl::shape_functions<specfem::element::dimension_tag::dim3>::
    shape_functions(const int ngllz, const int nglly, const int ngllx,
                    const int ngnod,
                    const specfem::assembly::mesh_impl::quadrature<
                        specfem::element::dimension_tag::dim3> &quadrature,
                    const specfem::assembly::mesh_impl::control_nodes<
                        specfem::element::dimension_tag::dim3>
                        control_nodes)
    : ngllz(ngllz), nglly(nglly), ngllx(ngllx), ngnod(control_nodes.ngnod),
      shape3D("specfem::assembly::shape_functions::shape3D", ngllz, nglly,
              ngllx, ngnod),
      dshape3D("specfem::assembly::shape_functions::dshape3D", ngllz, nglly,
               ngllx, ndim, ngnod),
      h_shape3D(Kokkos::create_mirror_view(shape3D)),
      h_dshape3D(Kokkos::create_mirror_view(dshape3D)) {

  const auto xi = quadrature.h_xi;

  Kokkos::parallel_for(
      "specfem::assembly::mesh::shape_functions::initialize_shape_functions",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<3> >({ 0, 0, 0 },
                                              { ngllz, nglly, ngllx }),
      [=, this](const int iz, const int iy, const int ix) {
        type_real xil = xi(ix);
        type_real etal = xi(iy);
        type_real zetal = xi(iz);
        const auto shape_function =
            specfem::shape_function::shape_function(xil, etal, zetal, ngnod);
        const auto shape_function_derivatives =
            specfem::shape_function::shape_function_derivatives(xil, etal,
                                                                zetal, ngnod);
        for (int in = 0; in < ngnod; in++) {
          h_shape3D(iz, iy, ix, in) = shape_function[in];
          h_dshape3D(iz, iy, ix, 0, in) = shape_function_derivatives[0][in];
          h_dshape3D(iz, iy, ix, 1, in) = shape_function_derivatives[1][in];
          h_dshape3D(iz, iy, ix, 2, in) = shape_function_derivatives[2][in];
        }
      });

  Kokkos::deep_copy(shape3D, h_shape3D);
  Kokkos::deep_copy(dshape3D, h_dshape3D);

  return;
}
