#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/property/impl/coordinates.hpp"
#include "specfem/io/property/impl/sub_block.hpp"
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
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
          ATTENUATION_SET(none, constant_isotropic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;
        constexpr auto attenuation_tag = TagsType::attenuation_tag;
        using AttenuationType = specfem::assembly::Attenuation<
            specfem::element::dimension_tag::dim2>;
        constexpr bool has_attenuation =
            AttenuationType::template has_attenuation<medium_tag, property_tag,
                                                      attenuation_tag>();

        const auto element_range = assembly.element_types.get_elements_on_host(
            medium_tag, property_tag, attenuation_tag);
        if (element_range.size() == 0)
          return;
        const std::string name =
            std::string("/") +
            specfem::element::to_string(medium_tag, property_tag,
                                        attenuation_tag);
        typename OutputLibrary::Group group = file.createGroup(name);
        const auto &data_container =
            assembly.properties.template get_container<medium_tag,
                                                       property_tag>();

        // The property container spans the whole (medium, property) group;
        // this combination's elements are the contiguous sub-block
        // [offset, offset + count) of its views.
        const int offset = element_range.begin_index() -
                           data_container.element_range.begin_index();
        const int count = element_range.size();

        if constexpr (has_attenuation) {
          const auto &att =
              assembly.attenuation.template get_container<medium_tag,
                                                          property_tag>();
          if (att.element_range.begin_index() != element_range.begin_index() ||
              att.element_range.size() != element_range.size()) {
            throw std::runtime_error(
                "property writer: attenuation container element range does "
                "not match the mesh element range for group " +
                name);
          }
        }

        property_impl::write_coordinates(group, assembly.mesh, element_range);

        // Property datasets. The file stores the physical (relaxed) moduli:
        // for attenuating combinations the staged "kappa"/"mu" values are
        // divided by the per-element scale factors before writing.
        data_container.for_each_host_view(
            [&](const auto view, const std::string view_name) {
              auto scratch = specfem::io::property_impl::extract_sub_block(
                  view, view_name, offset, count);
              if constexpr (has_attenuation) {
                assembly.attenuation
                    .template get_container<medium_tag, property_tag>()
                    .to_physical(scratch, view_name);
              }
              group.createDataset(view_name, scratch).write();
            });

        // Attenuation model datasets (Qkappa/Qmu), persisted verbatim.
        if constexpr (has_attenuation) {
          for (const auto &[view_name, view] :
               assembly.attenuation
                   .template get_container<medium_tag, property_tag>()
                   .get_views()) {
            group.createDataset(view_name, view).write();
          }
        }
        n_written += count;
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
          PROPERTY_SET(isotropic) * ATTENUATION_SET(none, constant_isotropic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;
        constexpr auto attenuation_tag = TagsType::attenuation_tag;
        using AttenuationType = specfem::assembly::Attenuation<
            specfem::element::dimension_tag::dim3>;
        constexpr bool has_attenuation =
            AttenuationType::template has_attenuation<medium_tag, property_tag,
                                                      attenuation_tag>();

        const auto element_range = assembly.element_types.get_elements_on_host(
            medium_tag, property_tag, attenuation_tag);
        if (element_range.size() == 0)
          return;
        const std::string name =
            std::string("/") +
            specfem::element::to_string(medium_tag, property_tag,
                                        attenuation_tag);
        typename OutputLibrary::Group group = file.createGroup(name);
        const auto &data_container =
            assembly.properties.template get_container<medium_tag,
                                                       property_tag>();

        // The property container spans the whole (medium, property) group;
        // this combination's elements are the contiguous sub-block
        // [offset, offset + count) of its views.
        const int offset = element_range.begin_index() -
                           data_container.element_range.begin_index();
        const int count = element_range.size();

        if constexpr (has_attenuation) {
          const auto &att =
              assembly.attenuation.template get_container<medium_tag,
                                                          property_tag>();
          if (att.element_range.begin_index() != element_range.begin_index() ||
              att.element_range.size() != element_range.size()) {
            throw std::runtime_error(
                "property writer: attenuation container element range does "
                "not match the mesh element range for group " +
                name);
          }
        }

        property_impl::write_coordinates(group, assembly.mesh, element_range);

        // Property datasets. The file stores the physical (relaxed) moduli:
        // for attenuating combinations the staged "kappa"/"mu" values are
        // divided by the per-element scale factors before writing.
        data_container.for_each_host_view(
            [&](const auto view, const std::string view_name) {
              auto scratch = specfem::io::property_impl::extract_sub_block(
                  view, view_name, offset, count);
              if constexpr (has_attenuation) {
                assembly.attenuation
                    .template get_container<medium_tag, property_tag>()
                    .to_physical(scratch, view_name);
              }
              group.createDataset(view_name, scratch).write();
            });

        // Attenuation model datasets (Qkappa/Qmu), persisted verbatim.
        if constexpr (has_attenuation) {
          for (const auto &[view_name, view] :
               assembly.attenuation
                   .template get_container<medium_tag, property_tag>()
                   .get_views()) {
            group.createDataset(view_name, view).write();
          }
        }
        n_written += count;
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
