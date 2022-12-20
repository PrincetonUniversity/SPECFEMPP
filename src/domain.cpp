#include "../include/domain.h"
#include "../include/compute.h"
#include "../include/config.h"
#include "../include/quadrature.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

specfem::Domain::Elastic::Elastic(
    const int ndim, const int nglob, specfem::compute::compute *compute,
    specfem::compute::properties *material_properties,
    specfem::compute::partial_derivatives *partial_derivatives,
    quadrature::quadrature *quadx, quadrature::quadrature *quadz)
    : field(specfem::HostView2d<type_real>("specfem::Domain::Elastic::field",
                                           nglob, ndim)),
      field_dot(specfem::HostView2d<type_real>(
          "specfem::Domain::Elastic::field_dot", nglob, ndim)),
      field_dot_dot(specfem::HostView2d<type_real>(
          "specfem::Domain::Elastic::field_dot_dot", nglob, ndim)),
      rmass_inverse(specfem::HostView2d<type_real>(
          "specfem::Domain::Elastic::rmass_inverse", nglob, ndim)),
      compute(compute), material_properties(material_properties),
      partial_derivatives(partial_derivatives), quadx(quadx), quadz(quadz) {

  const specfem::HostView3d<int> ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);
  // Initialize views
  Kokkos::parallel_for("specfem::Domain::Elastic::initiaze_views",
                       specfem::HostMDrange<2>({ 0, 0 }, { nglob, ndim }),
                       [=](const int iglob, const int idim) {
                         field(iglob, idim) = 0;
                         field_dot(iglob, idim) = 0;
                         field_dot_dot(iglob, idim) = 0;
                         rmass_inverse(iglob, idim) = 0;
                       });

  Kokkos::fence();
  // Compute the mass matrix
  Kokkos::Experimental::ScatterView<type_real **> results(rmass_inverse);
  auto wxgll = quadx->get_hw();
  auto wzgll = quadz->get_hw();
  Kokkos::parallel_for(
      "specfem::Domain::Elastic",
      specfem::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      [=](const int ispec, const int iz, const int ix) {
        int iglob = ibool(ispec, iz, ix);
        type_real rhol = material_properties->rho(ispec, iz, ix);
        auto access = results.access();
        if (material_properties->ispec_type(ispec) == elastic) {
          access(iglob, 0) += wxgll(ix) * wzgll(iz) * rhol *
                              partial_derivatives->jacobian(ispec, iz, ix);
          access(iglob, 1) += wxgll(ix) * wzgll(iz) * rhol *
                              partial_derivatives->jacobian(ispec, iz, ix);
        }
      });

  Kokkos::Experimental::contribute(rmass_inverse, results);
  Kokkos::fence();

  // invert the mass matrix
  Kokkos::parallel_for("specfem::Domain::Elastic", specfem::HostRange(0, nglob),
                       [=](const int iglob) {
                         if (rmass_inverse(iglob, 0) > 0.0) {
                           rmass_inverse(iglob, 0) =
                               1.0 / rmass_inverse(iglob, 0);
                           rmass_inverse(iglob, 1) =
                               1.0 / rmass_inverse(iglob, 1);
                         } else {
                           rmass_inverse(iglob, 0) = 1.0;
                           rmass_inverse(iglob, 1) = 1.0;
                         }
                       });

  return;
};
