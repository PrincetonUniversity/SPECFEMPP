#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/wavefield/writer.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/utilities.hpp"
#include <boost/filesystem.hpp>

template <typename OutputLibrary>
specfem::io::wavefield_writer<OutputLibrary>::wavefield_writer(
    const std::string &output_folder, const bool save_boundary_values,
    const bool save_attenuation_value)
    : output_folder(output_folder), save_boundary_values(save_boundary_values),
      save_attenuation_value(save_attenuation_value),
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
          boost::filesystem::create_directories(p.parent_path());
          return (p.parent_path() / p.stem()).string();
        }
        return p.string();
      }()),
      file(typename OutputLibrary::File(file_path)) {}

template <typename OutputLibrary>
template <specfem::element::dimension_tag DimensionTag>
void specfem::io::wavefield_writer<OutputLibrary>::initialize(
    specfem::assembly::assembly<DimensionTag> &assembly) {
  auto &forward = assembly.fields.forward;
  auto &mesh = assembly.mesh;
  auto &element_types = assembly.element_types;

  using DomainView =
      Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  int ngroups = 0;
  auto count_group = [&]<typename TagsType>() {
    if (forward.template get_nglob<TagsType::medium_tag>() > 0)
      ngroups++;
  };

  specfem::tag_dispatch::for_each(
      std::remove_reference_t<decltype(forward)>::combinations, count_group);

  Kokkos::View<std::string *, Kokkos::HostSpace> medium_tags("medium_tags",
                                                             ngroups);

  typename OutputLibrary::Group base_group =
      file.createGroup(std::string("/Coordinates"));

  int igroup = 0;

  auto process_medium = [&]<typename TagsType>() {
    constexpr auto medium_tag = TagsType::medium_tag;
    int nglob_medium = forward.template get_nglob<medium_tag>();
    if (nglob_medium > 0) {
      medium_tags(igroup) = specfem::element::to_string(medium_tag);
      igroup++;

      typename OutputLibrary::Group group =
          base_group.createGroup(specfem::element::to_string(medium_tag));

      const auto element_indices =
          element_types.get_elements_on_host(medium_tag);
      const int n_elements = element_indices.size();

      DomainView x("xcoordinates", nglob_medium);
      DomainView z("zcoordinates", nglob_medium);

      const int ngllz = mesh.element_grid.ngllz;
      const int ngllx = mesh.element_grid.ngllx;

      if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
        using MappingView =
            Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
        MappingView mapping("mapping", n_elements, ngllz, ngllx);
        for (int iel = 0; iel < n_elements; iel++) {
          const int ispec = element_indices(iel);
          for (int iz = 0; iz < ngllz; iz++) {
            for (int ix = 0; ix < ngllx; ix++) {
              const int iglob =
                  forward.template get_iglob<false, medium_tag>(ispec, iz, ix);
              mapping(iel, iz, ix) = iglob;
              x(iglob) = mesh.h_coord(0, ispec, iz, ix);
              z(iglob) = mesh.h_coord(1, ispec, iz, ix);
            }
          }
        }
        group.createDataset("X", x).write();
        group.createDataset("Z", z).write();
        group.createDataset("mapping", mapping).write();
      } else {
        const int nglly = mesh.element_grid.nglly;
        using MappingView =
            Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>;
        DomainView y("ycoordinates", nglob_medium);
        MappingView mapping("mapping", n_elements, ngllz, nglly, ngllx);
        for (int iel = 0; iel < n_elements; iel++) {
          const int ispec = element_indices(iel);
          for (int iz = 0; iz < ngllz; iz++) {
            for (int iy = 0; iy < nglly; iy++) {
              for (int ix = 0; ix < ngllx; ix++) {
                const int iglob = forward.template get_iglob<false, medium_tag>(
                    ispec, iz, iy, ix);
                mapping(iel, iz, iy, ix) = iglob;
                x(iglob) = mesh.h_coord(ispec, iz, iy, ix, 0);
                y(iglob) = mesh.h_coord(ispec, iz, iy, ix, 1);
                z(iglob) = mesh.h_coord(ispec, iz, iy, ix, 2);
              }
            }
          }
        }
        group.createDataset("X", x).write();
        group.createDataset("Y", y).write();
        group.createDataset("Z", z).write();
        group.createDataset("mapping", mapping).write();
      }
    }
  };

  specfem::tag_dispatch::for_each(
      std::remove_reference_t<decltype(forward)>::combinations, process_medium);

  file.createDataset("medium_tags", medium_tags).write();
  file.flush();

  specfem::Logger::info("Coordinates written to " + file_path);
}

template <typename OutputLibrary>
template <specfem::element::dimension_tag DimensionTag>
void specfem::io::wavefield_writer<OutputLibrary>::run(
    specfem::assembly::assembly<DimensionTag> &assembly, const int istep) {
  // When attenuation checkpointing is enabled, kinematic fields AND attenuation
  // memory variables are written in a single pass. This mirrors
  // save_forward_arrays_undoatt() from the Fortran code:
  //   R_trace (= Rkappa), R_xx, R_yy, R_xy, R_xz, R_yz are checkpointed.
  // SPECFEM++ also checkpoints the previous-step symmetrized strain fields used
  // by its SLS recurrence (epsilon_*_att); this is C++ restart state, not
  // kernel strain storage.
  //
  // We must write everything in one pass because write-mode I/O backends (NPY,
  // ASCII) only expose createGroup, not openGroup: we cannot create the step
  // group here and re-open it later to append attenuation datasets.
  auto &forward = assembly.fields.forward;
  auto &attenuation = assembly.attenuation;

  forward.copy_to_host();
  if (save_attenuation_value) {
    attenuation.copy_to_host();
  }

  typename OutputLibrary::Group step_group = file.createGroup(
      std::string("/Step") + specfem::utilities::to_zero_lead(istep, 6));

  // Write kinematic fields for every medium, and (when enabled) append the
  // attenuation datasets for those media that have attenuating elements.
  auto write_medium = [&]<typename TagsType>() {
    constexpr auto medium_tag = TagsType::medium_tag;
    const int nglob_medium = forward.template get_nglob<medium_tag>();
    if (nglob_medium > 0) {
      const auto &field = forward.template get_field<medium_tag>();
      typename OutputLibrary::Group med_group =
          step_group.createGroup(specfem::element::to_string(medium_tag));

      // Kinematic fields
      if constexpr (medium_tag == specfem::element::medium_tag::acoustic) {
        med_group.createDataset("Potential", field.get_host_field()).write();
        med_group.createDataset("PotentialDot", field.get_host_field_dot())
            .write();
        med_group
            .createDataset("PotentialDotDot", field.get_host_field_dot_dot())
            .write();
      } else {
        med_group.createDataset("Displacement", field.get_host_field()).write();
        med_group.createDataset("Velocity", field.get_host_field_dot()).write();
        med_group.createDataset("Acceleration", field.get_host_field_dot_dot())
            .write();
      }

      // Attenuation memory variables for this medium (if any)
      if (save_attenuation_value) {
        specfem::tag_dispatch::for_each(
            attenuation.attenuation_medium_combinations,
            [&]<typename AttTagsType>() {
              if constexpr (AttTagsType::medium_tag == medium_tag) {
                auto &medium =
                    attenuation.attenuation_storage.template get<AttTagsType>();
                if (medium.element_range.extent(0) == 0)
                  return;

                typename OutputLibrary::Group att_group =
                    med_group.createGroup("Attenuation");
                att_group.createDataset("Rkappa", medium.h_memory_variable_kappa)
                    .write();
                att_group.createDataset("Rxx", medium.h_memory_variable_Rxx)
                    .write();
                att_group.createDataset("Rxz", medium.h_memory_variable_Rxz)
                    .write();
                att_group.createDataset("EpsilonXX", medium.h_epsilon_xx_att)
                    .write();
                att_group.createDataset("EpsilonZZ", medium.h_epsilon_zz_att)
                    .write();
                att_group.createDataset("EpsilonXZ", medium.h_epsilon_xz_att)
                    .write();
                if constexpr (DimensionTag ==
                              specfem::element::dimension_tag::dim3) {
                  att_group.createDataset("Ryy", medium.h_memory_variable_Ryy)
                      .write();
                  att_group.createDataset("Rzz", medium.h_memory_variable_Rzz)
                      .write();
                  att_group.createDataset("Rxy", medium.h_memory_variable_Rxy)
                      .write();
                  att_group.createDataset("Ryz", medium.h_memory_variable_Ryz)
                      .write();
                  att_group.createDataset("EpsilonYY", medium.h_epsilon_yy_att)
                      .write();
                  att_group.createDataset("EpsilonXY", medium.h_epsilon_xy_att)
                      .write();
                  att_group.createDataset("EpsilonYZ", medium.h_epsilon_yz_att)
                      .write();
                }
              }
            });
      }
    }
  };

  specfem::tag_dispatch::for_each(
      std::remove_reference_t<decltype(forward)>::combinations, write_medium);

  file.flush();
}

template <typename OutputLibrary>
template <specfem::element::dimension_tag DimensionTag>
void specfem::io::wavefield_writer<OutputLibrary>::finalize(
    specfem::assembly::assembly<DimensionTag> &assembly) {

  typename OutputLibrary::Group boundary_group =
      file.createGroup(std::string("/BoundaryValues"));

  Kokkos::View<bool *, Kokkos::HostSpace> boundary_values_view(
      "save_boundary_values", 1);
  boundary_values_view(0) = this->save_boundary_values;
  boundary_group.createDataset("save_boundary_values", boundary_values_view)
      .write();

  if (save_boundary_values) {
    auto &boundary_values = assembly.boundary_values;
    boundary_values.copy_to_host();

    typename OutputLibrary::Group stacey = boundary_group.createGroup("Stacey");

    stacey
        .createDataset("IndexMapping",
                       boundary_values.stacey.h_property_index_mapping)
        .write();

    auto write_stacey = [&]<typename TagsType>() {
      constexpr auto medium_tag = TagsType::medium_tag;
      auto &ctr = boundary_values.stacey.container.template get<TagsType>();
      if (ctr.h_values.size() > 0) {
        const std::string dataset_name =
            specfem::element::to_string(medium_tag) + "Acceleration";
        stacey.createDataset(dataset_name, ctr.h_values).write();
      }
    };

    specfem::tag_dispatch::for_each(
        decltype(boundary_values.stacey)::combinations_by_medium, write_stacey);

    typename OutputLibrary::Group composite =
        boundary_group.createGroup("CompositeStaceyDirichlet");

    composite
        .createDataset(
            "IndexMapping",
            boundary_values.composite_stacey_dirichlet.h_property_index_mapping)
        .write();

    auto write_composite = [&]<typename TagsType>() {
      constexpr auto medium_tag = TagsType::medium_tag;
      auto &ctr = boundary_values.composite_stacey_dirichlet.container
                      .template get<TagsType>();
      if (ctr.h_values.size() > 0) {
        const std::string dataset_name =
            specfem::element::to_string(medium_tag) + "Acceleration";
        composite.createDataset(dataset_name, ctr.h_values).write();
      }
    };

    specfem::tag_dispatch::for_each(
        decltype(boundary_values
                     .composite_stacey_dirichlet)::combinations_by_medium,
        write_composite);

    file.flush();
  }

  specfem::Logger::info("Wavefield written to " + file_path);
}
