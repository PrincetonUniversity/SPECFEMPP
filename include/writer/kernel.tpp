#ifndef _SPECFEM_WRITER_KERNEL_TPP
#define _SPECFEM_WRITER_KERNEL_TPP

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/kernels.hpp"
#include "writer/kernel.hpp"
#include <Kokkos_Core.hpp>

template <typename OutputLibrary>
specfem::writer::kernel<OutputLibrary>::kernel(
    const specfem::compute::assembly &assembly, const std::string output_folder)
    : output_folder(output_folder), mesh(assembly.mesh),
      kernels(assembly.kernels) {}

template <typename OutputLibrary>
void specfem::writer::kernel<OutputLibrary>::write() {

  using DomainView =
      Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  kernels.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/Kernels");

  typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");

  const int nspec = mesh.points.nspec;
  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  int n_elastic_isotropic = 0;
  int n_elastic_anisotropic = 0;
  int n_acoustic = 0;

  Kokkos::parallel_reduce(
      "specfem::writer::kernel", specfem::kokkos::HostRange(0, nspec),
      [=](const int ispec, int &n_elastic_isotropic, int &n_elastic_anisotropic, int &n_acoustic) {
        if (kernels.h_element_types(ispec) ==
            specfem::element::medium_tag::elastic &&
            kernels.h_element_property(ispec) ==
            specfem::element::property_tag::isotropic) {
          n_elastic_isotropic++;
        } else if (kernels.h_element_types(ispec) ==
            specfem::element::medium_tag::elastic &&
            kernels.h_element_property(ispec) ==
            specfem::element::property_tag::anisotropic) {
          n_elastic_anisotropic++;
        } else if (kernels.h_element_types(ispec) ==
                   specfem::element::medium_tag::acoustic) {
          n_acoustic++;
        }
      },
      n_elastic_isotropic, n_elastic_anisotropic, n_acoustic);

  assert(n_elastic_isotropic + n_elastic_anisotropic + n_acoustic == nspec);

  {
    typename OutputLibrary::Group elastic = file.createGroup("/ElasticIsotropic");

    DomainView x("xcoordinates_elastic_isotropic", n_elastic_isotropic, ngllz, ngllx);
    DomainView z("zcoordinates_elastic_isotropic", n_elastic_isotropic, ngllz, ngllx);

    DomainView rho("rho", n_elastic_isotropic, ngllz, ngllx);
    DomainView mu("mu", n_elastic_isotropic, ngllz, ngllx);
    DomainView kappa("kappa", n_elastic_isotropic, ngllz, ngllx);
    DomainView rhop("rhop", n_elastic_isotropic, ngllz, ngllx);
    DomainView alpha("alpha", n_elastic_isotropic, ngllz, ngllx);
    DomainView beta("beta", n_elastic_isotropic, ngllz, ngllx);

    int i = 0;

    for (int ispec = 0; ispec < nspec; ispec++) {
      if (kernels.h_element_types(ispec) ==
          specfem::element::medium_tag::elastic) {
        for (int iz = 0; iz < ngllz; iz++) {
          for (int ix = 0; ix < ngllx; ix++) {
            x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
            z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
            const specfem::point::index<specfem::dimension::type::dim2> index(
                ispec, iz, ix);
            specfem::point::kernels<specfem::dimension::type::dim2,
                                    specfem::element::medium_tag::elastic,
                                    specfem::element::property_tag::isotropic,
                                    false>
                point_kernels;

            specfem::compute::load_on_host(index, kernels, point_kernels);

            rho(i, iz, ix) = point_kernels.rho;
            mu(i, iz, ix) = point_kernels.mu;
            kappa(i, iz, ix) = point_kernels.kappa;
            rhop(i, iz, ix) = point_kernels.rhop;
            alpha(i, iz, ix) = point_kernels.alpha;
            beta(i, iz, ix) = point_kernels.beta;
          }
        }
        i++;
      }
    }

    elastic.createDataset("X", x).write();
    elastic.createDataset("Z", z).write();
    elastic.createDataset("rho", rho).write();
    elastic.createDataset("mu", mu).write();
    elastic.createDataset("kappa", kappa).write();
    elastic.createDataset("rhop", rhop).write();
    elastic.createDataset("alpha", alpha).write();
    elastic.createDataset("beta", beta).write();
  }

  {
    typename OutputLibrary::Group elastic = file.createGroup("/ElasticAnisotropic");

    DomainView x("xcoordinates_elastic_anisotropic", n_elastic_anisotropic, ngllz, ngllx);
    DomainView z("zcoordinates_elastic_anisotropic", n_elastic_anisotropic, ngllz, ngllx);

    DomainView rho("rho", n_elastic_anisotropic, ngllz, ngllx);
    DomainView c11("c11", n_elastic_anisotropic, ngllz, ngllx);
    DomainView c13("c13", n_elastic_anisotropic, ngllz, ngllx);
    DomainView c15("c15", n_elastic_anisotropic, ngllz, ngllx);
    DomainView c33("c33", n_elastic_anisotropic, ngllz, ngllx);
    DomainView c35("c35", n_elastic_anisotropic, ngllz, ngllx);
    DomainView c55("c55", n_elastic_anisotropic, ngllz, ngllx);

    int i = 0;

    for (int ispec = 0; ispec < nspec; ispec++) {
      if (kernels.h_element_types(ispec) ==
          specfem::element::medium_tag::elastic) {
        for (int iz = 0; iz < ngllz; iz++) {
          for (int ix = 0; ix < ngllx; ix++) {
            x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
            z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
            const specfem::point::index<specfem::dimension::type::dim2> index(
                ispec, iz, ix);
            specfem::point::kernels<specfem::dimension::type::dim2,
                                    specfem::element::medium_tag::elastic,
                                    specfem::element::property_tag::anisotropic,
                                    false>
                point_kernels;

            specfem::compute::load_on_host(index, kernels, point_kernels);

            rho(i, iz, ix) = point_kernels.rho;
            c11(i, iz, ix) = point_kernels.c11;
            c13(i, iz, ix) = point_kernels.c13;
            c15(i, iz, ix) = point_kernels.c15;
            c33(i, iz, ix) = point_kernels.c33;
            c35(i, iz, ix) = point_kernels.c35;
            c55(i, iz, ix) = point_kernels.c55;
          }
        }
        i++;
      }
    }

    elastic.createDataset("X", x).write();
    elastic.createDataset("Z", z).write();
    elastic.createDataset("rho", rho).write();
    elastic.createDataset("c11", c11).write();
    elastic.createDataset("c13", c13).write();
    elastic.createDataset("c15", c15).write();
    elastic.createDataset("c33", c33).write();
    elastic.createDataset("c35", c35).write();
    elastic.createDataset("c55", c55).write();
  }

  {
    DomainView x("xcoordinates_acoustic", n_acoustic, ngllz, ngllx);
    DomainView z("zcoordinates_acoustic", n_acoustic, ngllz, ngllx);

    DomainView rho("rho", n_acoustic, ngllz, ngllx);
    DomainView kappa("kappa", n_acoustic, ngllz, ngllx);
    DomainView rho_prime("rho_prime", n_acoustic, ngllz, ngllx);
    DomainView alpha("alpha", n_acoustic, ngllz, ngllx);

    int i = 0;

    for (int ispec = 0; ispec < nspec; ispec++) {
      if (kernels.h_element_types(ispec) ==
          specfem::element::medium_tag::acoustic) {
        for (int iz = 0; iz < ngllz; iz++) {
          for (int ix = 0; ix < ngllx; ix++) {
            x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
            z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
            const specfem::point::index<specfem::dimension::type::dim2> index(
                ispec, iz, ix);
            specfem::point::kernels<specfem::dimension::type::dim2,
                                    specfem::element::medium_tag::acoustic,
                                    specfem::element::property_tag::isotropic,
                                    false>
                point_kernels;

            specfem::compute::load_on_host(index, kernels, point_kernels);

            rho(i, iz, ix) = point_kernels.rho;
            kappa(i, iz, ix) = point_kernels.kappa;
            rho_prime(i, iz, ix) = point_kernels.rhop;
            alpha(i, iz, ix) = point_kernels.alpha;
          }
        }
        i++;
      }
    }

    acoustic.createDataset("X", x).write();
    acoustic.createDataset("Z", z).write();
    acoustic.createDataset("rho", rho).write();
    acoustic.createDataset("kappa", kappa).write();
    acoustic.createDataset("rho_prime", rho_prime).write();
    acoustic.createDataset("alpha", alpha).write();
  }

  std::cout << "Kernels written to " << output_folder << "/Kernels"
            << std::endl;
}

#endif /* _SPECFEM_WRITER_KERNEL_TPP */
