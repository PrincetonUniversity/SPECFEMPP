#pragma once

#include "specfem/assembly.hpp"
#include "domain_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "io/impl/medium_writer.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

template <typename OutputLibrary, typename ContainerType>
void specfem::io::impl::write_container(
    const std::string &output_folder, const std::string &output_namespace,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::element_types<specfem::dimension::type::dim2> &element_types,
    ContainerType &container) {
  using DomainView =
      specfem::kokkos::DomainView2d<type_real, 3, Kokkos::HostSpace>;

  container.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/" + output_namespace);

  const int nspec = mesh.nspec;
  const int ngllz = mesh.element_grid.ngllz;
  const int ngllx = mesh.element_grid.ngllx;

  int n_written = 0;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      {
        const std::string name =
            std::string("/") +
            specfem::element::to_string(_medium_tag_, _property_tag_);
        typename OutputLibrary::Group group = file.createGroup(name);
        const auto element_indices =
            element_types.get_elements_on_host(_medium_tag_, _property_tag_);
        const int n_elements = element_indices.size();
        n_written += n_elements;
        DomainView x("xcoordinates", n_elements, ngllz, ngllx);
        DomainView z("zcoordinates", n_elements, ngllz, ngllx);
        for (int i = 0; i < n_elements; i++) {
          const int ispec = element_indices(i);
          for (int iz = 0; iz < ngllz; iz++) {
            for (int ix = 0; ix < ngllx; ix++) {
              x(i, iz, ix) = mesh.h_coord(0, ispec, iz, ix);
              z(i, iz, ix) = mesh.h_coord(1, ispec, iz, ix);
            }
          }
        }
        group.createDataset("X", x).write();
        group.createDataset("Z", z).write();
        auto data_container =
            container.template get_container<_medium_tag_, _property_tag_>();

        data_container.for_each_host_view(
            [&](const auto view, const std::string name) mutable {
              group.createDataset(name, view).write();
            });
      })

  if (n_written != nspec) {
    std::ostringstream message;
    message << "Error while writing output container at" << __FILE__ << ":"
            << __LINE__ << "\n"
            << "Error writing output: expected to write " << nspec
            << " elements, but wrote " << n_written << " elements.";
    throw std::runtime_error(message.str());
  }

  std::cout << output_namespace << " written to " << output_folder << "/"
            << output_namespace << std::endl;
}
