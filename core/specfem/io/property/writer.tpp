#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/impl/element_group.hpp"
#include "specfem/io/property/writer.hpp"
#include "specfem/io/xdmf/derived_quantities.hpp"
#include "specfem/io/xdmf/description.hpp"
#include "specfem/io/xdmf/writer.hpp"
#include "specfem/io_backends/HDF5/HDF5.hpp"
#include "specfem/logger.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/mpi.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template <typename OutputLibrary>
specfem::io::property_writer<OutputLibrary>::property_writer(
    const std::string &output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void specfem::io::property_writer<OutputLibrary>::write(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly) {
  write_impl(assembly);
}

template <typename OutputLibrary>
void specfem::io::property_writer<OutputLibrary>::write(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  write_impl(assembly);
}

template <typename OutputLibrary>
template <specfem::element::dimension_tag DimensionTag>
void specfem::io::property_writer<OutputLibrary>::write_impl(
    specfem::assembly::assembly<DimensionTag> &assembly) {
  // Build rank-specific output path following the same convention as mesh
  // files:
  //   serial:   {output_folder}/Properties/
  //   parallel: {output_folder}/Properties/proc_N/
  const boost::filesystem::path path =
      specfem::MPI::format_proc_filename(output_folder + "/Properties");

  // Ensure intermediate directories exist (needed for MPI case where
  // the parent output_folder/Properties/ may not exist yet)
  boost::filesystem::create_directories(path.parent_path());

  assembly.properties.copy_to_host();

  typename OutputLibrary::File file(path.string());

  const int nspec = assembly.mesh.nspec;

  int n_written = 0;

  std::vector<specfem::io::xdmf::ElementGroupDescription<DimensionTag> >
      descriptions;

  specfem::tag_dispatch::for_each(
      assembly.properties.combinations *
          ATTENUATION_SET(none, constant_isotropic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;
        constexpr auto attenuation_tag = TagsType::attenuation_tag;
        using AttenuationType = specfem::assembly::Attenuation<DimensionTag>;
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

        // Property datasets. The property container spans the whole (medium,
        // property) group; this combination's elements are a contiguous
        // sub-block of its views. The file stores the physical (relaxed)
        // moduli: for attenuating combinations the staged "kappa"/"mu" values
        // are divided by the per-element scale factors before writing.
        specfem::io::impl::write_element_group(
            group, assembly.mesh, element_range, data_container,
            [&](auto &scratch, const std::string &view_name) {
              if constexpr (has_attenuation) {
                assembly.attenuation
                    .template get_container<medium_tag, property_tag>()
                    .to_physical(scratch, view_name);
              }
            });

        specfem::io::xdmf::ElementGroupDescription<DimensionTag> description;
        description.group = name;
        description.number_of_elements = element_range.size();
        description.element_grid = assembly.mesh.element_grid;
        const auto extents = specfem::io::xdmf::element_extents(
            description.number_of_elements, description.element_grid);
        data_container.for_each_host_view(
            [&](const auto &, const std::string &view_name) {
              description.datasets.push_back({ view_name, extents });
            });

        // Attenuation model datasets (Qkappa/Qmu), persisted verbatim.
        if constexpr (has_attenuation) {
          for (const auto &[view_name, view] :
               assembly.attenuation
                   .template get_container<medium_tag, property_tag>()
                   .get_views()) {
            group.createDataset(view_name, view).write();
            description.datasets.push_back({ view_name, extents });
          }
        }
        description.derived =
            specfem::io::xdmf::DerivedQuantities<medium_tag,
                                                 property_tag>::list();
        descriptions.push_back(description);
        n_written += element_range.size();
      });

  if (n_written != nspec) {
    std::ostringstream message;
    message << "Error while writing output container at" << __FILE__ << ":"
            << __LINE__ << "\n"
            << "Error writing output: expected to write " << nspec
            << " elements, but wrote " << n_written << " elements.";
    throw std::runtime_error(message.str());
  }

  if constexpr (std::is_same_v<
                    OutputLibrary,
                    specfem::io_backends::HDF5<specfem::io::write> >) {
    specfem::io::xdmf::write_element_groups(path.string() + ".xmf",
                                            path.filename().string() + ".h5",
                                            descriptions);
  }

  specfem::Logger::info("Properties written to " + path.string());
}
