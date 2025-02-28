#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/kernels.hpp"
#include "IO/kernel/writer.hpp"
#include <Kokkos_Core.hpp>

template <typename OutputLibrary>
specfem::IO::kernel_writer<OutputLibrary>::kernel_writer(const std::string output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void specfem::IO::kernel_writer<OutputLibrary>::write(specfem::compute::assembly &assembly) {
  const auto &mesh = assembly.mesh;
  auto &element_types = assembly.element_types;
  auto &kernels = assembly.kernels;

  using DomainView =
      Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  using DomainKernelView =
      Kokkos::View<type_real ****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  kernels.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/Kernels");

  const int nspec = mesh.points.nspec;
  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  int n_elastic_isotropic;
  int n_elastic_anisotropic;
  int n_acoustic;

  {
    typename OutputLibrary::Group elastic = file.createGroup("/ElasticIsotropic");

    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic);
    n_elastic_isotropic = element_indices.size();

    DomainView x("xcoordinates_elastic_isotropic", n_elastic_isotropic, ngllz, ngllx);
    DomainView z("zcoordinates_elastic_isotropic", n_elastic_isotropic, ngllz, ngllx);

    using PointKernelType = typename specfem::point::kernels<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic,
                            false>;
    constexpr int nprops = PointKernelType::nprops;

    DomainKernelView data("data", n_elastic_isotropic, ngllz, ngllx, nprops);

    for (int i = 0; i < n_elastic_isotropic; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
          z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
          const specfem::point::index<specfem::dimension::type::dim2> index(
              ispec, iz, ix);
          PointKernelType point_kernels;

          specfem::compute::load_on_host(index, kernels, point_kernels);

          for (int l = 0; l < nprops; l++) {
            data(i, iz, ix, l) = point_kernels.data[l];
          }
        }
      }
    }

    elastic.createDataset("X", x).write();
    elastic.createDataset("Z", z).write();
    elastic.createDataset("data", data).write();
  }

  {
    typename OutputLibrary::Group elastic = file.createGroup("/ElasticAnisotropic");

    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic);
    n_elastic_anisotropic = element_indices.size();

    DomainView x("xcoordinates_elastic_anisotropic", n_elastic_anisotropic, ngllz, ngllx);
    DomainView z("zcoordinates_elastic_anisotropic", n_elastic_anisotropic, ngllz, ngllx);

    using PointKernelType = typename specfem::point::kernels<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::anisotropic,
                            false>;
    constexpr int nprops = PointKernelType::nprops;

    DomainKernelView data("data", n_elastic_anisotropic, ngllz, ngllx, nprops);

    for (int i = 0; i < n_elastic_anisotropic; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
          z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
          const specfem::point::index<specfem::dimension::type::dim2> index(
              ispec, iz, ix);
          PointKernelType point_kernels;

          specfem::compute::load_on_host(index, kernels, point_kernels);

          for (int l = 0; l < nprops; l++) {
            data(i, iz, ix, l) = point_kernels.data[l];
          }
        }
      }
    }

    elastic.createDataset("X", x).write();
    elastic.createDataset("Z", z).write();
    elastic.createDataset("data", data).write();
  }

  {
    typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");

    const auto element_indices = element_types.get_elements_on_host(specfem::element::medium_tag::acoustic);
    n_acoustic = element_indices.size();

    DomainView x("xcoordinates_acoustic", n_acoustic, ngllz, ngllx);
    DomainView z("zcoordinates_acoustic", n_acoustic, ngllz, ngllx);

    using PointKernelType = typename specfem::point::kernels<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic,
                            false>;
    constexpr int nprops = PointKernelType::nprops;

    DomainKernelView data("data", n_acoustic, ngllz, ngllx, nprops);

    for (int i = 0; i < n_acoustic; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, ix) = mesh.points.h_coord(0, ispec, iz, ix);
          z(i, iz, ix) = mesh.points.h_coord(1, ispec, iz, ix);
          const specfem::point::index<specfem::dimension::type::dim2> index(
              ispec, iz, ix);
          PointKernelType point_kernels;

          specfem::compute::load_on_host(index, kernels, point_kernels);

          for (int l = 0; l < nprops; l++) {
            data(i, iz, ix, l) = point_kernels.data[l];
          }
        }
      }
    }

    acoustic.createDataset("X", x).write();
    acoustic.createDataset("Z", z).write();
    acoustic.createDataset("data", data).write();
  }

  assert(n_elastic_isotropic + n_elastic_anisotropic + n_acoustic == nspec);

  std::cout << "Kernels written to " << output_folder << "/Kernels"
            << std::endl;
}
