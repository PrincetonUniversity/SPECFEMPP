#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/impl/medium_writer.hpp"
#include "specfem/io/property/writer.hpp"
#include "specfem/logger.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/mpi.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

template <typename OutputLibrary>
specfem::io::property_writer<OutputLibrary>::property_writer(
    const std::string &output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void specfem::io::property_writer<OutputLibrary>::write(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly) {
  // Build rank-specific output path following the same convention as mesh
  // files:
  //   serial:   {output_folder}/Properties/
  //   parallel: {output_folder}/Properties/proc_N/
  // format_proc_filename("foo.ext") -> "foo/proc_N.ext" (nproc>1) or "foo.ext"
  // (nproc==1)
  const std::string formatted =
      specfem::MPI::format_proc_filename(output_folder + "/Properties");
  const boost::filesystem::path formatted_path(formatted);
  const std::string base_folder = formatted_path.parent_path().string();
  const std::string ns = formatted_path.stem().string();

  // Ensure intermediate directories exist (needed for MPI case where
  // base_folder = output_folder/Properties/ which may not exist yet)
  boost::filesystem::create_directories(base_folder);

  assembly.properties.copy_to_host();

  typename OutputLibrary::File file(base_folder + "/" + ns);

  const int nspec = assembly.mesh.nspec;

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
            assembly.element_types.get_elements_on_host(medium_tag,
                                                        property_tag);
        if (element_indices.size() == 0)
          return;
        const std::string name =
            std::string("/") +
            specfem::element::to_string(medium_tag, property_tag);
        typename OutputLibrary::Group group = file.createGroup(name);
        const auto data_container =
            assembly.properties.template get_container<medium_tag,
                                                       property_tag>();

        impl::write_coordinates(group, assembly.mesh, element_indices);

        using AttenuationType = specfem::assembly::Attenuation<
            specfem::element::dimension_tag::dim2>;
        if constexpr (AttenuationType::template has_attenuation<
                          medium_tag, property_tag>()) {
          // The model file stores PHYSICAL (relaxed) moduli plus the
          // attenuation model datasets (e.g. Qkappa/Qmu); which views are
          // transformed is decided by the attenuation container.
          const auto &att =
              assembly.attenuation.template get_container<medium_tag,
                                                          property_tag>();
          data_container.for_each_host_view(
              [&](const auto view, const std::string view_name) {
                group
                    .createDataset(view_name, att.physical_view(
                                                  view, view_name,
                                                  element_indices))
                    .write();
              });
          att.for_each_io_host_view(
              [&](const auto view, const std::string view_name) {
                group.createDataset(view_name, view).write();
              });
        } else {
          data_container.for_each_host_view(
              [&](const auto view, const std::string view_name) {
                group.createDataset(view_name, view).write();
              });
        }
        n_written += element_indices.size();
      });

  if (n_written != nspec) {
    std::ostringstream message;
    message << "Error while writing output container at" << __FILE__ << ":"
            << __LINE__ << "\n"
            << "Error writing output: expected to write " << nspec
            << " elements, but wrote " << n_written << " elements.";
    throw std::runtime_error(message.str());
  }

  specfem::Logger::info(ns + " written to " + base_folder + "/" + ns);
}

template <typename OutputLibrary>
void specfem::io::property_writer<OutputLibrary>::write(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  // Build rank-specific output path following the same convention as mesh
  // files:
  //   serial:   {output_folder}/Properties/
  //   parallel: {output_folder}/Properties/proc_N/
  const std::string formatted =
      specfem::MPI::format_proc_filename(output_folder + "/Properties");
  const boost::filesystem::path formatted_path(formatted);
  const std::string base_folder = formatted_path.parent_path().string();
  const std::string ns = formatted_path.stem().string();

  boost::filesystem::create_directories(base_folder);

  assembly.properties.copy_to_host();

  typename OutputLibrary::File file(base_folder + "/" + ns);

  const int nspec = assembly.mesh.nspec;

  int n_written = 0;

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(elastic, acoustic) *
          PROPERTY_SET(isotropic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;

        const auto element_indices =
            assembly.element_types.get_elements_on_host(medium_tag,
                                                        property_tag);
        if (element_indices.size() == 0)
          return;
        const std::string name =
            std::string("/") +
            specfem::element::to_string(medium_tag, property_tag);
        typename OutputLibrary::Group group = file.createGroup(name);
        const auto data_container =
            assembly.properties.template get_container<medium_tag,
                                                       property_tag>();

        impl::write_coordinates(group, assembly.mesh, element_indices);

        using AttenuationType = specfem::assembly::Attenuation<
            specfem::element::dimension_tag::dim3>;
        if constexpr (AttenuationType::template has_attenuation<
                          medium_tag, property_tag>()) {
          // The model file stores PHYSICAL (relaxed) moduli plus the
          // attenuation model datasets (e.g. Qkappa/Qmu); which views are
          // transformed is decided by the attenuation container.
          const auto &att =
              assembly.attenuation.template get_container<medium_tag,
                                                          property_tag>();
          data_container.for_each_host_view(
              [&](const auto view, const std::string view_name) {
                group
                    .createDataset(view_name, att.physical_view(
                                                  view, view_name,
                                                  element_indices))
                    .write();
              });
          att.for_each_io_host_view(
              [&](const auto view, const std::string view_name) {
                group.createDataset(view_name, view).write();
              });
        } else {
          data_container.for_each_host_view(
              [&](const auto view, const std::string view_name) {
                group.createDataset(view_name, view).write();
              });
        }
        n_written += element_indices.size();
      });

  if (n_written != nspec) {
    std::ostringstream message;
    message << "Error while writing output container at" << __FILE__ << ":"
            << __LINE__ << "\n"
            << "Error writing output: expected to write " << nspec
            << " elements, but wrote " << n_written << " elements.";
    throw std::runtime_error(message.str());
  }

  specfem::Logger::info(ns + " written to " + base_folder + "/" + ns);
}
