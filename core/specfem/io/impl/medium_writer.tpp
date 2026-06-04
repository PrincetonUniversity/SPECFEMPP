#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/io/impl/medium_writer.hpp"
#include "specfem/logger.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

template <typename GroupType, typename ElementIndicesType,
          typename DataContainerType>
int specfem::io::impl::write_medium_group(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const ElementIndicesType &element_indices,
    const DataContainerType &data_container) {

  const int ngllz = mesh.element_grid.ngllz;
  const int ngllx = mesh.element_grid.ngllx;

  using DomainView =
      specfem::datatype::DomainView2d<type_real, 3, Kokkos::HostSpace>;

  const int n_elements = element_indices.size();
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

  data_container.for_each_host_view(
      [&](const auto view, const std::string name) mutable {
        group.createDataset(name, view).write();
      });

  return n_elements;
}

template <typename OutputLibrary, typename ContainerType>
void specfem::io::impl::write_container(
    const std::string &output_folder, const std::string &output_namespace,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &element_types,
    ContainerType &container) {
  using DomainView =
      specfem::datatype::DomainView2d<type_real, 3, Kokkos::HostSpace>;

  container.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/" + output_namespace);

  const int nspec = mesh.nspec;

  int n_written = 0;

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;

        const auto element_indices =
            element_types.get_elements_on_host(medium_tag, property_tag);
        if (element_indices.size() == 0)
          return;
        const std::string name =
            std::string("/") + specfem::element::to_string(medium_tag,
                                                           property_tag);
        typename OutputLibrary::Group group = file.createGroup(name);
        auto data_container =
            container.template get_container<medium_tag, property_tag>();
        n_written +=
            write_medium_group(group, mesh, element_indices, data_container);
      });

  if (n_written != nspec) {
    std::ostringstream message;
    message << "Error while writing output container at" << __FILE__ << ":"
            << __LINE__ << "\n"
            << "Error writing output: expected to write " << nspec
            << " elements, but wrote " << n_written << " elements.";
    throw std::runtime_error(message.str());
  }

  specfem::Logger::info(output_namespace + " written to " + output_folder +
                        "/" + output_namespace);
}

// ---------------------------------------------------------------------------
// dim3 overloads
// ---------------------------------------------------------------------------

template <typename GroupType, typename ElementIndicesType,
          typename DataContainerType>
int specfem::io::impl::write_medium_group(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const ElementIndicesType &element_indices,
    const DataContainerType &data_container) {

  const int ngllz = mesh.element_grid.ngllz;
  const int nglly = mesh.element_grid.nglly;
  const int ngllx = mesh.element_grid.ngllx;

  const int n_elements = element_indices.size();

  using DomainView3d =
      specfem::datatype::DomainView<specfem::element::dimension_tag::dim3,
                                    type_real, 4, Kokkos::HostSpace>;
  DomainView3d x("xcoordinates", n_elements, ngllz, nglly, ngllx);
  DomainView3d y("ycoordinates", n_elements, ngllz, nglly, ngllx);
  DomainView3d z("zcoordinates", n_elements, ngllz, nglly, ngllx);

  for (int i = 0; i < n_elements; i++) {
    const int ispec = element_indices(i);
    for (int iz = 0; iz < ngllz; iz++) {
      for (int iy = 0; iy < nglly; iy++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, iy, ix) = mesh.h_coord(ispec, iz, iy, ix, 0);
          y(i, iz, iy, ix) = mesh.h_coord(ispec, iz, iy, ix, 1);
          z(i, iz, iy, ix) = mesh.h_coord(ispec, iz, iy, ix, 2);
        }
      }
    }
  }
  group.createDataset("X", x).write();
  group.createDataset("Y", y).write();
  group.createDataset("Z", z).write();

  data_container.for_each_host_view(
      [&](const auto view, const std::string name) mutable {
        group.createDataset(name, view).write();
      });

  return n_elements;
}

template <typename OutputLibrary, typename ContainerType>
void specfem::io::impl::write_container(
    const std::string &output_folder, const std::string &output_namespace,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &element_types,
    ContainerType &container) {

  container.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/" + output_namespace);

  const int nspec = mesh.nspec;

  int n_written = 0;

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(elastic, acoustic) *
          PROPERTY_SET(isotropic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;

        const auto element_indices =
            element_types.get_elements_on_host(medium_tag, property_tag);
        if (element_indices.size() == 0)
          return;
        const std::string name =
            std::string("/") + specfem::element::to_string(medium_tag,
                                                           property_tag);
        typename OutputLibrary::Group group = file.createGroup(name);
        auto data_container =
            container.template get_container<medium_tag, property_tag>();
        n_written +=
            write_medium_group(group, mesh, element_indices, data_container);
      });

  if (n_written != nspec) {
    std::ostringstream message;
    message << "Error while writing output container at" << __FILE__ << ":"
            << __LINE__ << "\n"
            << "Error writing output: expected to write " << nspec
            << " elements, but wrote " << n_written << " elements.";
    throw std::runtime_error(message.str());
  }

  specfem::Logger::info(output_namespace + " written to " + output_folder +
                        "/" + output_namespace);
}
