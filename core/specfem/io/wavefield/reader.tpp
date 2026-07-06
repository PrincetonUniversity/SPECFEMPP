#pragma once

#include "specfem/assembly.hpp"
#include "specfem/io/wavefield/reader.hpp"
#include "specfem/mpi.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/utilities.hpp"
#include <boost/filesystem.hpp>

template <typename IOLibrary>
specfem::io::wavefield_reader<IOLibrary>::wavefield_reader(
    const std::string &output_folder, const bool load_attenuation_value)
    : output_folder(output_folder),
      load_attenuation_value(load_attenuation_value),
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
      // File is opened lazily in open_file() / initialize() so that the reader
      // can be constructed before the checkpoint directory exists (needed for
      // the combined_undoatt workflow where the forward pass creates the dir).
      file(std::nullopt) {}

template <typename IOLibrary>
void specfem::io::wavefield_reader<IOLibrary>::open_file() {
  if (!file.has_value()) {
    file.emplace(file_path);
  }
}

template <typename IOLibrary>
template <specfem::element::dimension_tag DimensionTag>
void specfem::io::wavefield_reader<IOLibrary>::initialize(
    specfem::assembly::assembly<DimensionTag> &assembly) {

  auto &buffer = assembly.fields.buffer;
  int ngroups = 0;

  auto count_group = [&]<typename TagsType>() {
    if (buffer.template get_nglob<TagsType::medium_tag>() > 0)
      ngroups++;
  };

  specfem::tag_dispatch::for_each(
      std::remove_reference_t<decltype(buffer)>::combinations, count_group);

  open_file();

  Kokkos::View<std::string *, Kokkos::HostSpace> medium_tags("medium_tags",
                                                             ngroups);
  file->openDataset("medium_tags", medium_tags).read();

  auto check_medium = [&]<typename TagsType>() {
    constexpr auto medium_tag = TagsType::medium_tag;
    if (buffer.template get_nglob<medium_tag>() > 0) {
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
  };

  specfem::tag_dispatch::for_each(
      std::remove_reference_t<decltype(buffer)>::combinations, check_medium);

  auto &boundary_values = assembly.boundary_values;

  typename IOLibrary::Group boundary_group = file->openGroup("/BoundaryValues");

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

  auto read_stacey = [&]<typename TagsType>() {
    constexpr auto medium_tag = TagsType::medium_tag;
    auto &ctr = boundary_values.stacey.container.template get<TagsType>();
    if (ctr.h_values.size() > 0) {
      const std::string dataset_name =
          specfem::element::to_string(medium_tag) + "Acceleration";
      stacey.openDataset(dataset_name, ctr.h_values).read();
    }
  };

  specfem::tag_dispatch::for_each(
      decltype(boundary_values.stacey)::combinations_by_medium, read_stacey);

  typename IOLibrary::Group composite =
      boundary_group.openGroup("CompositeStaceyDirichlet");

  composite
      .openDataset(
          "IndexMapping",
          boundary_values.composite_stacey_dirichlet.h_property_index_mapping)
      .read();

  auto read_composite = [&]<typename TagsType>() {
    constexpr auto medium_tag = TagsType::medium_tag;
    auto &ctr = boundary_values.composite_stacey_dirichlet.container
                    .template get<TagsType>();
    if (ctr.h_values.size() > 0) {
      const std::string dataset_name =
          specfem::element::to_string(medium_tag) + "Acceleration";
      composite.openDataset(dataset_name, ctr.h_values).read();
    }
  };

  specfem::tag_dispatch::for_each(
      decltype(boundary_values
                   .composite_stacey_dirichlet)::combinations_by_medium,
      read_composite);

  boundary_values.copy_to_device();
}

template <typename IOLibrary>
template <specfem::element::dimension_tag DimensionTag>
void specfem::io::wavefield_reader<IOLibrary>::run(
    specfem::assembly::assembly<DimensionTag> &assembly, const int istep) {
  open_file();
  auto &buffer = assembly.fields.buffer;

  typename IOLibrary::Group base_group = file->openGroup(
      std::string("/Step") + specfem::utilities::to_zero_lead(istep, 6));

  auto read_field = [&]<typename TagsType>() {
    constexpr auto medium_tag = TagsType::medium_tag;
    int nglob_medium = buffer.template get_nglob<medium_tag>();
    if (nglob_medium > 0) {
      typename IOLibrary::Group group =
          base_group.openGroup(specfem::element::to_string(medium_tag));
      const auto &field = buffer.template get_field<medium_tag>();
      if constexpr (medium_tag == specfem::element::medium_tag::acoustic) {
        group.openDataset("Potential", field.get_host_field()).read();
        group.openDataset("PotentialDot", field.get_host_field_dot()).read();
        group.openDataset("PotentialDotDot", field.get_host_field_dot_dot())
            .read();
      } else {
        group.openDataset("Displacement", field.get_host_field()).read();
        group.openDataset("Velocity", field.get_host_field_dot()).read();
        group.openDataset("Acceleration", field.get_host_field_dot_dot())
            .read();
      }
    }
  };

  specfem::tag_dispatch::for_each(
      std::remove_reference_t<decltype(buffer)>::combinations, read_field);

  buffer.copy_to_device();

  if (!load_attenuation_value) {
    return;
  }

  // Load attenuation memory variables into assembly.attenuation.
  // The combined_undoatt solver will deep_copy these into its forward
  // attenuation container before replaying the subset, mirroring
  // read_forward_arrays_undoatt() in the Fortran code.
  auto &attenuation = assembly.attenuation;

  specfem::tag_dispatch::for_each(
      attenuation.attenuation_medium_combinations, [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;

        auto &medium = attenuation.attenuation_storage.template get<TagsType>();

        if (medium.element_range.extent(0) == 0)
          return;

        typename IOLibrary::Group med_group =
            base_group.openGroup(specfem::element::to_string(medium_tag));
        typename IOLibrary::Group att_group =
            med_group.openGroup("Attenuation");

        att_group.openDataset("Rkappa", medium.h_memory_variable_kappa).read();
        att_group.openDataset("Rxx", medium.h_memory_variable_Rxx).read();
        att_group.openDataset("Rxz", medium.h_memory_variable_Rxz).read();
        att_group.openDataset("EpsilonXX", medium.h_epsilon_xx_att).read();
        att_group.openDataset("EpsilonZZ", medium.h_epsilon_zz_att).read();
        att_group.openDataset("EpsilonXZ", medium.h_epsilon_xz_att).read();

        if constexpr (DimensionTag == specfem::element::dimension_tag::dim3) {
          att_group.openDataset("Ryy", medium.h_memory_variable_Ryy).read();
          att_group.openDataset("Rzz", medium.h_memory_variable_Rzz).read();
          att_group.openDataset("Rxy", medium.h_memory_variable_Rxy).read();
          att_group.openDataset("Ryz", medium.h_memory_variable_Ryz).read();
          att_group.openDataset("EpsilonYY", medium.h_epsilon_yy_att).read();
          att_group.openDataset("EpsilonXY", medium.h_epsilon_xy_att).read();
          att_group.openDataset("EpsilonYZ", medium.h_epsilon_yz_att).read();
        }
      });

  attenuation.copy_to_device();
}
