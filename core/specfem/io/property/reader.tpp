#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/io/property/impl/sub_block.hpp"
#include "specfem/io/property/reader.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"

#include "specfem/mpi.hpp"
#include "specfem/point.hpp"
#include <algorithm>
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
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
          ATTENUATION_SET(none, constant_isotropic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;
        constexpr auto attenuation_tag = TagsType::attenuation_tag;

        // Skip combinations with no elements — the writer never created
        // a group for them, so there is nothing to read.
        const auto element_range = assembly.element_types.get_elements_on_host(
            medium_tag, property_tag, attenuation_tag);
        if (element_range.size() == 0)
          return;

        const std::string name =
            std::string("/") +
            specfem::element::to_string(medium_tag, property_tag,
                                        attenuation_tag);
        typename InputLibrary::Group group = file.openGroup(name);
        const auto &container =
            properties.get_container<medium_tag, property_tag>();

        // The property container spans the whole (medium, property) group;
        // this combination's datasets cover the contiguous sub-block
        // [offset, offset + count) of its views.
        const int offset = element_range.begin_index() -
                           container.element_range.begin_index();
        const int count = element_range.size();

        // Datasets are serialized in logical row-major order (see
        // sub_block.hpp), so every read is staged through a plain scratch
        // view and unpacked into the chunk-tiled container view.
        const auto read_sub_block = [&](const auto view,
                                        const std::string view_name) {
          auto scratch = specfem::io::property_impl::make_sub_block(
              view, view_name, count);
          group.openDataset(view_name, scratch).read();
          specfem::io::property_impl::insert_sub_block(view, scratch, offset);
        };

        using AttenuationType = specfem::assembly::Attenuation<
            specfem::element::dimension_tag::dim2>;
        if constexpr (AttenuationType::template has_attenuation<
                          medium_tag, property_tag, attenuation_tag>()) {
          const auto &att =
              assembly.attenuation.template get_container<medium_tag,
                                                          property_tag>();
          // The attenuation container owns the "kappa"/"mu" datasets (the
          // reference physical moduli) plus the attenuation model datasets
          // (Qkappa/Qmu); all are read verbatim. The remaining property
          // views are read for this combination's sub-block. recompute then
          // derives the runtime state (unrelaxed moduli, relaxation rates)
          // from the on-disk model.
          const auto io_views = att.io_views();
          container.for_each_host_view(
              [&](const auto view, const std::string view_name) {
                const bool owned_by_attenuation =
                    std::any_of(io_views.begin(), io_views.end(),
                                [&](const auto &entry) {
                                  return entry.first == view_name;
                                });
                if (owned_by_attenuation)
                  return;
                read_sub_block(view, view_name);
              });
          for (const auto &[view_name, view] : io_views) {
            auto scratch = specfem::io::property_impl::make_sub_block(
                view, view_name, static_cast<int>(view.extent(0)));
            group.openDataset(view_name, scratch).read();
            specfem::io::property_impl::insert_sub_block(view, scratch, 0);
          }
          att.recompute(container, assembly.attenuation.fc,
                        assembly.attenuation.f0, assembly.attenuation.band,
                        assembly.attenuation.tau_sigma);
        } else {
          container.for_each_host_view(
              [&](const auto view, const std::string view_name) {
                read_sub_block(view, view_name);
              });
        }
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

  // Build rank-specific input path following the same convention as mesh
  // files:
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
          PROPERTY_SET(isotropic) *
          ATTENUATION_SET(none, constant_isotropic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        constexpr auto property_tag = TagsType::property_tag;
        constexpr auto attenuation_tag = TagsType::attenuation_tag;

        // Skip combinations with no elements — the writer never created
        // a group for them, so there is nothing to read.
        const auto element_range = assembly.element_types.get_elements_on_host(
            medium_tag, property_tag, attenuation_tag);
        if (element_range.size() == 0)
          return;

        const std::string name =
            std::string("/") +
            specfem::element::to_string(medium_tag, property_tag,
                                        attenuation_tag);
        typename InputLibrary::Group group = file.openGroup(name);
        const auto &container =
            properties.get_container<medium_tag, property_tag>();

        // The property container spans the whole (medium, property) group;
        // this combination's datasets cover the contiguous sub-block
        // [offset, offset + count) of its views.
        const int offset = element_range.begin_index() -
                           container.element_range.begin_index();
        const int count = element_range.size();

        // Datasets are serialized in logical row-major order (see
        // sub_block.hpp), so every read is staged through a plain scratch
        // view and unpacked into the chunk-tiled container view.
        const auto read_sub_block = [&](const auto view,
                                        const std::string view_name) {
          auto scratch = specfem::io::property_impl::make_sub_block(
              view, view_name, count);
          group.openDataset(view_name, scratch).read();
          specfem::io::property_impl::insert_sub_block(view, scratch, offset);
        };

        using AttenuationType = specfem::assembly::Attenuation<
            specfem::element::dimension_tag::dim3>;
        if constexpr (AttenuationType::template has_attenuation<
                          medium_tag, property_tag, attenuation_tag>()) {
          const auto &att =
              assembly.attenuation.template get_container<medium_tag,
                                                          property_tag>();
          // The attenuation container owns the "kappa"/"mu" datasets (the
          // reference physical moduli) plus the attenuation model datasets
          // (Qkappa/Qmu); all are read verbatim. The remaining property
          // views are read for this combination's sub-block. recompute then
          // derives the runtime state (unrelaxed moduli, relaxation rates)
          // from the on-disk model.
          const auto io_views = att.io_views();
          container.for_each_host_view(
              [&](const auto view, const std::string view_name) {
                const bool owned_by_attenuation =
                    std::any_of(io_views.begin(), io_views.end(),
                                [&](const auto &entry) {
                                  return entry.first == view_name;
                                });
                if (owned_by_attenuation)
                  return;
                read_sub_block(view, view_name);
              });
          for (const auto &[view_name, view] : io_views) {
            auto scratch = specfem::io::property_impl::make_sub_block(
                view, view_name, static_cast<int>(view.extent(0)));
            group.openDataset(view_name, scratch).read();
            specfem::io::property_impl::insert_sub_block(view, scratch, 0);
          }
          att.recompute(container, assembly.attenuation.fc,
                        assembly.attenuation.f0, assembly.attenuation.band,
                        assembly.attenuation.tau_sigma);
        } else {
          container.for_each_host_view(
              [&](const auto view, const std::string view_name) {
                read_sub_block(view, view_name);
              });
        }
      });

  std::cout << "Properties read from " << base_folder << "/" << ns
            << std::endl;

  properties.copy_to_device();
}
