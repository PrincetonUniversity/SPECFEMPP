#ifndef _SPECFEM_WRITER_KERNEL_TPP
#define _SPECFEM_WRITER_KERNEL_TPP

#include "kokkos_abstractions.h"
#include "writer/kernel.hpp"
#include <Kokkos_Core.hpp>

template <typename OutputLibrary>
specfem::writer::kernel<OutputLibrary>::kernel(
    const specfem::compute::assembly &assembly, const std::string output_folder)
    : output_folder(output_folder), mesh(assembly.mesh),
      kernels(assembly.kernels) {}

template <typename OutputLibrary>
void specfem::writer::kernel<OutputLibrary>::write() {

  kernels.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/Kernels");

  typename OutputLibrary::Group elastic = file.createGroup("/Elastic");
  typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");

  const int nspec = mesh.points.nspec;
  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  int nelastic = 0;
  int nacoustic = 0;

  Kokkos::parallel_reduce(
      "specfem::writer::kernel", specfem::kokkos::HostRange(0, nspec),
      [=](const int ispec, int &nelastic, int &nacoustic) {
        if (kernels.h_element_types(ispec) ==
            specfem::element::medium_tag::elastic) {
          nelastic++;
        } else if (kernels.h_element_types(ispec) ==
                   specfem::element::medium_tag::acoustic) {
          nacoustic++;
        }
      },
      nelastic, nacoustic);

  assert(nelastic + nacoustic == nspec);

  specfem::kokkos::HostView3d<type_real> xcoordinates_elastic(
      "xcoordinates_elastic", nelastic, ngllz, ngllx);
  specfem::kokkos::HostView3d<type_real> zcoordinates_elastic(
      "zcoordinates_elastic", nelastic, ngllz, ngllx);

  int index = 0;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if (kernels.h_element_types(ispec) ==
        specfem::element::medium_tag::elastic) {
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          xcoordinates_elastic(index, iz, ix) =
              mesh.points.h_coord(0, ispec, iz, ix);
          zcoordinates_elastic(index, iz, ix) =
              mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
      index++;
    }
  }

  elastic.createDataset("X", xcoordinates_elastic).write();
  elastic.createDataset("Z", zcoordinates_elastic).write();
  elastic.createDataset("rho", kernels.elastic_isotropic.h_rho).write();
  elastic.createDataset("mu", kernels.elastic_isotropic.h_mu).write();
  elastic.createDataset("kappa", kernels.elastic_isotropic.h_kappa).write();
  elastic.createDataset("rhop", kernels.elastic_isotropic.h_rhop).write();
  elastic.createDataset("alpha", kernels.elastic_isotropic.h_alpha).write();
  elastic.createDataset("beta", kernels.elastic_isotropic.h_beta).write();

  specfem::kokkos::HostView3d<type_real> xcoordinates_acoustic(
      "xcoordinates_acoustic", nacoustic, ngllz, ngllx);
  specfem::kokkos::HostView3d<type_real> zcoordinates_acoustic(
      "zcoordinates_acoustic", nacoustic, ngllz, ngllx);

  index = 0;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if (kernels.h_element_types(ispec) ==
        specfem::element::medium_tag::acoustic) {
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          xcoordinates_acoustic(index, iz, ix) =
              mesh.points.h_coord(0, ispec, iz, ix);
          zcoordinates_acoustic(index, iz, ix) =
              mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
      index++;
    }
  }

  acoustic.createDataset("X", xcoordinates_acoustic).write();
  acoustic.createDataset("Z", zcoordinates_acoustic).write();
  acoustic.createDataset("rho", kernels.acoustic_isotropic.h_rho).write();
  acoustic.createDataset("kappa", kernels.acoustic_isotropic.h_kappa).write();
  acoustic.createDataset("rho_prime", kernels.acoustic_isotropic.h_rho_prime)
      .write();
  acoustic.createDataset("alpha", kernels.acoustic_isotropic.h_alpha).write();

  std::cout << "Kernels written to " << output_folder << "/Kernels"
            << std::endl;
}

#endif /* _SPECFEM_WRITER_KERNEL_TPP */
