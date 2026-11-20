#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/kernel/writer.hpp"
#include "specfem/io/impl/element_group.hpp"
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
specfem::io::kernel_writer<OutputLibrary>::kernel_writer(
    const std::string &output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void specfem::io::kernel_writer<OutputLibrary>::write(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly) {
  write_impl(assembly);
}

template <typename OutputLibrary>
void specfem::io::kernel_writer<OutputLibrary>::write(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  write_impl(assembly);
}

template <typename OutputLibrary>
template <specfem::element::dimension_tag DimensionTag>
void specfem::io::kernel_writer<OutputLibrary>::write_impl(
    specfem::assembly::assembly<DimensionTag> &assembly) {
  // Rank-specific output path: {output_folder}/Kernels[/proc_N]
  const boost::filesystem::path path =
      specfem::MPI::format_proc_filename(output_folder + "/Kernels");

  boost::filesystem::create_directories(path.parent_path());

  assembly.kernels.copy_to_host();

  typename OutputLibrary::File file(path.string());

  const int nspec = assembly.mesh.nspec;

  int n_written = 0;

  std::vector<specfem::io::xdmf::ElementGroupDescription<DimensionTag> >
      descriptions;

  specfem::tag_dispatch::for_each(
      assembly.kernels.combinations, [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;

        const auto element_range =
            assembly.element_types.get_elements_on_host(medium_tag,
                                                        property_tag);
        if (element_range.size() == 0)
          return;
        const std::string name =
            std::string("/") +
            specfem::element::to_string(medium_tag, property_tag);
        typename OutputLibrary::Group group = file.createGroup(name);
        const auto &container =
            assembly.kernels.template get_container<medium_tag, property_tag>();

        specfem::io::impl::write_element_group(group, assembly.mesh,
                                               element_range, container);

        specfem::io::xdmf::ElementGroupDescription<DimensionTag> description;
        description.group = name;
        description.number_of_elements = element_range.size();
        description.element_grid = assembly.mesh.element_grid;
        const auto extents = specfem::io::xdmf::element_extents(
            description.number_of_elements, description.element_grid);
        container.for_each_host_view(
            [&](const auto &, const std::string &view_name) {
              description.datasets.push_back({ view_name, extents });
            });
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

  specfem::Logger::info("Kernels written to " + path.string());
}
