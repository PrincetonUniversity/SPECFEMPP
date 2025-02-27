#pragma once

#include "IO/property/writer.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/properties.hpp"
#include <Kokkos_Core.hpp>

template <typename OutputLibrary>
specfem::IO::property_writer<OutputLibrary>::property_writer(
    const std::string output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void specfem::IO::property_writer<OutputLibrary>::write(
    specfem::compute::assembly &assembly) {
  const auto &mesh = assembly.mesh;
  auto &element_types = assembly.element_types;
  auto &properties = assembly.properties;

  using DomainView =
      Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  properties.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/Properties");

  const int nspec = mesh.points.nspec;
  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  int n_elastic_sv_isotropic;
  int n_elastic_sh_isotropic;
  int n_elastic_sv_anisotropic;
  int n_elastic_sh_anisotropic;
  int n_acoustic;

  {
    typename OutputLibrary::Group elastic =
        file.createGroup("/ElasticSVIsotropic");

    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic_sv,
        specfem::element::property_tag::isotropic);
    n_elastic_sv_isotropic = element_indices.size();

    DomainView x("xcoordinates_elastic_isotropic", n_elastic_sv_isotropic,
                 ngllz, ngllx);
    DomainView z("zcoordinates_elastic_isotropic", n_elastic_sv_isotropic,
                 ngllz, ngllx);

    for (int i = 0; i < n_elastic_sv_isotropic; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
          z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }

    elastic.createDataset("X", x).write();
    elastic.createDataset("Z", z).write();

    elastic
        .createDataset("rho", properties.value_dim2_elastic_sv_isotropic.h_rho)
        .write();
    elastic.createDataset("mu", properties.value_dim2_elastic_sv_isotropic.h_mu)
        .write();
    elastic
        .createDataset(
            "lambdaplus2mu",
            properties.value_dim2_elastic_sv_isotropic.h_lambdaplus2mu)
        .write();
  }

  {
    typename OutputLibrary::Group elastic =
        file.createGroup("/ElasticSHIsotropic");

    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic_sh,
        specfem::element::property_tag::isotropic);
    n_elastic_sh_isotropic = element_indices.size();

    DomainView x("xcoordinates_elastic_isotropic", n_elastic_sh_isotropic,
                 ngllz, ngllx);
    DomainView z("zcoordinates_elastic_isotropic", n_elastic_sh_isotropic,
                 ngllz, ngllx);

    for (int i = 0; i < n_elastic_sh_isotropic; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
          z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }

    elastic.createDataset("X", x).write();
    elastic.createDataset("Z", z).write();

    elastic
        .createDataset("rho", properties.value_dim2_elastic_sh_isotropic.h_rho)
        .write();
    elastic.createDataset("mu", properties.value_dim2_elastic_sh_isotropic.h_mu)
        .write();
    elastic
        .createDataset(
            "lambdaplus2mu",
            properties.value_dim2_elastic_sh_isotropic.h_lambdaplus2mu)
        .write();
  }

  {
    typename OutputLibrary::Group elastic =
        file.createGroup("/ElasticSVAnisotropic");

    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic_sv,
        specfem::element::property_tag::anisotropic);
    n_elastic_sv_anisotropic = element_indices.size();

    DomainView x("xcoordinates_elastic_anisotropic", n_elastic_sv_anisotropic,
                 ngllz, ngllx);
    DomainView z("zcoordinates_elastic_anisotropic", n_elastic_sv_anisotropic,
                 ngllz, ngllx);

    for (int i = 0; i < n_elastic_sv_anisotropic; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
          z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }

    elastic.createDataset("X", x).write();
    elastic.createDataset("Z", z).write();

    elastic
        .createDataset("rho",
                       properties.value_dim2_elastic_sv_anisotropic.h_rho)
        .write();
    elastic
        .createDataset("c11",
                       properties.value_dim2_elastic_sv_anisotropic.h_c11)
        .write();
    elastic
        .createDataset("c13",
                       properties.value_dim2_elastic_sv_anisotropic.h_c13)
        .write();
    elastic
        .createDataset("c15",
                       properties.value_dim2_elastic_sv_anisotropic.h_c15)
        .write();
    elastic
        .createDataset("c33",
                       properties.value_dim2_elastic_sv_anisotropic.h_c33)
        .write();
    elastic
        .createDataset("c35",
                       properties.value_dim2_elastic_sv_anisotropic.h_c35)
        .write();
    elastic
        .createDataset("c55",
                       properties.value_dim2_elastic_sv_anisotropic.h_c55)
        .write();
    elastic
        .createDataset("c12",
                       properties.value_dim2_elastic_sv_anisotropic.h_c12)
        .write();
    elastic
        .createDataset("c23",
                       properties.value_dim2_elastic_sv_anisotropic.h_c23)
        .write();
    elastic
        .createDataset("c25",
                       properties.value_dim2_elastic_sv_anisotropic.h_c25)
        .write();
  }

  {
    typename OutputLibrary::Group elastic =
        file.createGroup("/ElasticSHAnisotropic");

    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic_sh,
        specfem::element::property_tag::anisotropic);
    n_elastic_sh_anisotropic = element_indices.size();

    DomainView x("xcoordinates_elastic_anisotropic", n_elastic_sh_anisotropic,
                 ngllz, ngllx);
    DomainView z("zcoordinates_elastic_anisotropic", n_elastic_sh_anisotropic,
                 ngllz, ngllx);

    for (int i = 0; i < n_elastic_sh_anisotropic; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
          z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }

    elastic.createDataset("X", x).write();
    elastic.createDataset("Z", z).write();

    elastic
        .createDataset("rho",
                       properties.value_dim2_elastic_sh_anisotropic.h_rho)
        .write();
    elastic
        .createDataset("c11",
                       properties.value_dim2_elastic_sh_anisotropic.h_c11)
        .write();
    elastic
        .createDataset("c13",
                       properties.value_dim2_elastic_sh_anisotropic.h_c13)
        .write();
    elastic
        .createDataset("c15",
                       properties.value_dim2_elastic_sh_anisotropic.h_c15)
        .write();
    elastic
        .createDataset("c33",
                       properties.value_dim2_elastic_sh_anisotropic.h_c33)
        .write();
    elastic
        .createDataset("c35",
                       properties.value_dim2_elastic_sh_anisotropic.h_c35)
        .write();
    elastic
        .createDataset("c55",
                       properties.value_dim2_elastic_sh_anisotropic.h_c55)
        .write();
    elastic
        .createDataset("c12",
                       properties.value_dim2_elastic_sh_anisotropic.h_c12)
        .write();
    elastic
        .createDataset("c23",
                       properties.value_dim2_elastic_sh_anisotropic.h_c23)
        .write();
    elastic
        .createDataset("c25",
                       properties.value_dim2_elastic_sh_anisotropic.h_c25)
        .write();
  }

  {
    typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");

    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::acoustic);
    n_acoustic = element_indices.size();

    DomainView x("xcoordinates_acoustic", n_acoustic, ngllz, ngllx);
    DomainView z("zcoordinates_acoustic", n_acoustic, ngllz, ngllx);

    for (int i = 0; i < n_acoustic; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
          z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }

    acoustic.createDataset("X", x).write();
    acoustic.createDataset("Z", z).write();

    acoustic
        .createDataset("rho_inverse",
                       properties.value_dim2_acoustic_isotropic.h_rho_inverse)
        .write();
    acoustic
        .createDataset("kappa",
                       properties.value_dim2_acoustic_isotropic.h_kappa)
        .write();
  }

  assert(n_elastic_sv_isotropic + n_elastic_sv_anisotropic +
             n_elastic_sh_isotropic + n_elastic_sh_anisotropic + n_acoustic ==
         nspec);

  std::cout << "Properties written to " << output_folder << "/Properties"
            << std::endl;
}
