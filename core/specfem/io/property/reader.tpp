#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/io/impl/attenuation_io.hpp"
#include "specfem/io/property/reader.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"

#include "specfem/mpi.hpp"
#include "specfem/point.hpp"
#include <boost/filesystem.hpp>
#include <Kokkos_Core.hpp>

template <typename InputLibrary>
specfem::io::property_reader<InputLibrary>::property_reader(
    const std::string &input_folder)
    : input_folder(input_folder) {}

template <typename InputLibrary>
void specfem::io::property_reader<InputLibrary>::read(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly) {
  auto &properties = assembly.properties;

  // Build rank-specific input path following the same convention as mesh files:
  //   serial:   {input_folder}/Properties/
  //   parallel: {input_folder}/Properties/proc_N/
  const std::string formatted =
      specfem::MPI::format_proc_filename(input_folder + "/Properties");
  const boost::filesystem::path formatted_path(formatted);
  const std::string base_folder = formatted_path.parent_path().string();
  const std::string ns = formatted_path.stem().string();

  typename InputLibrary::File file(base_folder + "/" + ns);

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;

        // Skip combinations with no elements — the writer never created
        // a group for them, so there is nothing to read.
        const auto element_indices =
            assembly.element_types.get_elements_on_host(medium_tag,
                                                        property_tag);
        if (element_indices.size() == 0)
          return;

        const std::string name =
            std::string("/") +
            specfem::element::to_string(medium_tag, property_tag);
        typename InputLibrary::Group group = file.openGroup(name);
        const auto container =
            properties.get_container<medium_tag, property_tag>();
        const specfem::io::impl::AttenuationIO<
            decltype(assembly.attenuation), decltype(assembly.element_types)>
            att_io{ assembly.attenuation, assembly.element_types };
        // 1. Read the physical (relaxed) property views and Qkappa/Qmu.
        container.for_each_host_view(
            [&](const auto view, const std::string view_name) {
              group.openDataset(view_name, view).read();
            });
        // Below are no-op if the (medium, property) combination has no attenuation container.
        att_io.template attenuation_read<medium_tag, property_tag>(group);
        // 2. Recompute the modulus scale factors from the read-back Q.
        assembly.attenuation
            .template recompute_scaling<medium_tag, property_tag>();
        container.for_each_host_view(
            [&](const auto view, const std::string view_name) {
              att_io.template rescale_read<medium_tag, property_tag>(view,
                                                                     view_name);
            });
      });

  std::cout << "Properties read from " << base_folder << "/" << ns
            << std::endl;

  properties.copy_to_device();
}

template <typename InputLibrary>
void specfem::io::property_reader<InputLibrary>::read(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  auto &properties = assembly.properties;

  // Build rank-specific input path following the same convention as mesh files:
  //   serial:   {input_folder}/Properties/
  //   parallel: {input_folder}/Properties/proc_N/
  const std::string formatted =
      specfem::MPI::format_proc_filename(input_folder + "/Properties");
  const boost::filesystem::path formatted_path(formatted);
  const std::string base_folder = formatted_path.parent_path().string();
  const std::string ns = formatted_path.stem().string();

  typename InputLibrary::File file(base_folder + "/" + ns);

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(elastic, acoustic) *
          PROPERTY_SET(isotropic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;

        // Skip combinations with no elements — the writer never created
        // a group for them, so there is nothing to read.
        const auto element_indices =
            assembly.element_types.get_elements_on_host(medium_tag,
                                                        property_tag);
        if (element_indices.size() == 0)
          return;

        const std::string name =
            std::string("/") +
            specfem::element::to_string(medium_tag, property_tag);
        typename InputLibrary::Group group = file.openGroup(name);
        const auto container =
            properties.get_container<medium_tag, property_tag>();
        // 1. Read the physical (relaxed) property views and Qkappa/Qmu.
        container.for_each_host_view(
            [&](const auto view, const std::string view_name) {
              group.openDataset(view_name, view).read();
            });

        // Create helper struct to read attenuation values.
        const specfem::io::impl::AttenuationIO<
            decltype(assembly.attenuation), decltype(assembly.element_types)>
            att_io{ assembly.attenuation, assembly.element_types };

        att_io.template attenuation_read<medium_tag, property_tag>(group);
        assembly.attenuation
            .template recompute_scaling<medium_tag, property_tag>();
        container.for_each_host_view(
            [&](const auto view, const std::string view_name) {
              att_io.template rescale_read<medium_tag, property_tag>(view,
                                                                     view_name);
            });
      });

  std::cout << "Properties read from " << base_folder << "/" << ns
            << std::endl;

  properties.copy_to_device();
}
