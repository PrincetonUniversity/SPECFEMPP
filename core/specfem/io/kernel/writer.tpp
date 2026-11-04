#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/kernel/writer.hpp"
#include "specfem/io/impl/element_group.hpp"
#include "specfem/logger.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

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
  const auto path = specfem::io::impl::make_proc_path(output_folder, "Kernels");

  boost::filesystem::create_directories(path.base_folder);

  assembly.kernels.copy_to_host();

  typename OutputLibrary::File file(path.full());

  const int nspec = assembly.mesh.nspec;

  int n_written = 0;

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

        // The kernel container spans the whole (medium, property) group;
        // this combination's elements are the contiguous sub-block
        // [offset, offset + count) of its views.
        const int offset =
            element_range.begin_index() - container.element_range.begin_index();
        const int count = element_range.size();

        specfem::io::impl::write_element_group(group, assembly.mesh,
                                               element_range, container,
                                               offset, count);
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

  specfem::Logger::info(path.name + " written to " + path.full());
}
