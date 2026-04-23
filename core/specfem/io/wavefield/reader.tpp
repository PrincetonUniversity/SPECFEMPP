#pragma once

#include "specfem/assembly.hpp"
#include "specfem/io/wavefield/reader.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/utilities.hpp"
#include "specfem/mpi.hpp"
#include <boost/filesystem.hpp>

template <typename IOLibrary>
specfem::io::wavefield_reader<IOLibrary>::wavefield_reader(
    const std::string &output_folder)
    : output_folder(output_folder),
      // Build rank-specific path:
      //   serial:   {output_folder}/ForwardWavefield
      //   parallel: {output_folder}/ForwardWavefield/proc_N
      // file_path is declared before file in the header, so it is initialized
      // first and can safely be passed to the File constructor.
      file_path([&output_folder]() {
        const std::string formatted = specfem::MPI::format_proc_filename(
            output_folder + "/ForwardWavefield");
        const boost::filesystem::path p(formatted);
        if (specfem::MPI::get_size() > 1) {
          return (p.parent_path() / p.stem()).string();
        }
        return p.string();
      }()),
      file(typename IOLibrary::File(file_path)) {}

template <typename IOLibrary>
void specfem::io::wavefield_reader<IOLibrary>::initialize(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly) {

  auto &buffer = assembly.fields.buffer;
  int ngroups = 0;

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_psv_t, elastic_sh, acoustic,
                     poroelastic),
      [&]<typename TagsType>() {
        if (buffer.get_nglob<TagsType::medium_tag>() > 0) {
          ngroups++;
        }
      });

  Kokkos::View<std::string *, Kokkos::HostSpace> medium_tags("medium_tags", ngroups);
  file.openDataset("medium_tags", medium_tags).read();

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_psv_t, elastic_sh, acoustic,
                     poroelastic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        if (buffer.get_nglob<medium_tag>() > 0) {
          const std::string current_tag = specfem::element::to_string(medium_tag);
          bool found = false;
          for (int i = 0; i < (int)medium_tags.extent(0); ++i) {
            if (current_tag == medium_tags(i)) {
              found = true;
              break;
            }
          }
          if (!found) {
            throw std::runtime_error("Medium tag " + current_tag +
                                     " not found in wavefield file");
          }
        }
      });

  auto &boundary_values = assembly.boundary_values;

  typename IOLibrary::Group boundary_group = file.openGroup("/BoundaryValues");

  Kokkos::View<bool *, Kokkos::HostSpace> boundary_values_view(
      "save_boundary_values", 1);

  boundary_group.openDataset("save_boundary_values", boundary_values_view)
      .read();

  if (!boundary_values_view(0)) {
    throw std::runtime_error("Boundary values were not saved in the wavefield "
                             "output, please set `for_adjoint_simulations` to "
                             "true in the input file for forward simulations.");
  }

  typename IOLibrary::Group stacey = boundary_group.openGroup("Stacey");

  stacey
      .openDataset("IndexMapping",
                   boundary_values.stacey.h_property_index_mapping)
      .read();

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        auto &ctr = boundary_values.stacey.container.template get<TagsType>();
        if (ctr.h_values.size() > 0) {
          const std::string dataset_name =
              specfem::element::to_string(medium_tag) + "Acceleration";
          stacey.openDataset(dataset_name, ctr.h_values).read();
        }
      });

  boundary_values.copy_to_device();

  return;
}

template <typename IOLibrary>
void specfem::io::wavefield_reader<IOLibrary>::run(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly,
    const int istep) {
  auto &buffer = assembly.fields.buffer;

  // Note: boundary values not read/needed?
  // auto &boundary_values = assembly.boundary_values;

  typename IOLibrary::Group base_group = file.openGroup(
      std::string("/Step") + specfem::utilities::to_zero_lead(istep, 6));

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        int nglob_medium = buffer.get_nglob<medium_tag>();

        if (nglob_medium > 0) {
          typename IOLibrary::Group group =
              base_group.openGroup(specfem::element::to_string(medium_tag));
          const auto &field = buffer.get_field<medium_tag>();

          if constexpr (medium_tag == specfem::element::medium_tag::acoustic) {
            group.openDataset("Potential", field.get_host_field()).read();
            group.openDataset("PotentialDot", field.get_host_field_dot())
                .read();
            group.openDataset("PotentialDotDot", field.get_host_field_dot_dot())
                .read();
          } else {
            group.openDataset("Displacement", field.get_host_field()).read();
            group.openDataset("Velocity", field.get_host_field_dot()).read();
            group.openDataset("Acceleration", field.get_host_field_dot_dot())
                .read();
          }
        }
      });

  buffer.copy_to_device();
}
