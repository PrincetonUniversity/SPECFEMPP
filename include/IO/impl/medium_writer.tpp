#pragma once

#include "IO/impl/medium_writer.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

template <typename OutputLibrary, typename ContainerType>
void specfem::IO::impl::write_container(
    const std::string &output_folder, const std::string &output_namespace,
    const specfem::compute::mesh &mesh,
    const specfem::compute::element_types &element_types,
    ContainerType &container) {
  using DomainView =
      Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  container.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/" + output_namespace);

  const int nspec = mesh.points.nspec;
  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  int n_elastic_isotropic;
  int n_elastic_anisotropic;
  int n_acoustic;

  {
    typename OutputLibrary::Group elastic =
        file.createGroup("/ElasticIsotropic");

    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic);
    n_elastic_isotropic = element_indices.size();

    DomainView x("xcoordinates_elastic_isotropic", n_elastic_isotropic, ngllz,
                 ngllx);
    DomainView z("zcoordinates_elastic_isotropic", n_elastic_isotropic, ngllz,
                 ngllx);

    for (int i = 0; i < n_elastic_isotropic; i++) {
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
    elastic.createDataset("data", container.elastic_isotropic.h_data).write();
  }

  {
    typename OutputLibrary::Group elastic =
        file.createGroup("/ElasticAnisotropic");

    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic);
    n_elastic_anisotropic = element_indices.size();

    DomainView x("xcoordinates_elastic_anisotropic", n_elastic_anisotropic,
                 ngllz, ngllx);
    DomainView z("zcoordinates_elastic_anisotropic", n_elastic_anisotropic,
                 ngllz, ngllx);

    for (int i = 0; i < n_elastic_anisotropic; i++) {
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
    elastic.createDataset("data", container.elastic_anisotropic.h_data).write();
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
    acoustic.createDataset("data", container.acoustic_isotropic.h_data).write();
  }

  assert(n_elastic_isotropic + n_elastic_anisotropic + n_acoustic == nspec);

  std::cout << output_namespace << " written to " << output_folder << "/"
            << output_namespace << std::endl;
}
