#pragma once

#include "specfem/enums.hpp"
#include "specfem/io/wavefield/writer.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/utilities.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include <boost/filesystem.hpp>

template <typename OutputLibrary>
specfem::io::wavefield_writer<OutputLibrary>::wavefield_writer(
    const std::string &output_folder, const bool save_boundary_values)
    : output_folder(output_folder),
      save_boundary_values(save_boundary_values),
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
void specfem::io::wavefield_writer<OutputLibrary>::initialize(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly) {
  auto &forward = assembly.fields.forward;
  auto &mesh = assembly.mesh;
  auto &element_types = assembly.element_types;

  using DomainView =
      Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  using MappingView =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  const int ngllz = mesh.element_grid.ngllz;
  const int ngllx = mesh.element_grid.ngllx;

  int ngroups = 0;
  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_psv_t, elastic_sh, acoustic,
                     poroelastic),
      [&]<typename TagsType>() {
        if (forward.get_nglob<TagsType::medium_tag>() > 0) {
          ngroups++;
        }
      });

  Kokkos::View<std::string *, Kokkos::HostSpace> medium_tags("medium_tags", ngroups);

  typename OutputLibrary::Group base_group =
      file.createGroup(std::string("/Coordinates"));

  int igroup = 0;

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_psv_t, elastic_sh, acoustic,
                     poroelastic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;

        // Get the number of GLL points in the medium
        int nglob_medium = forward.get_nglob<medium_tag>();

        if (nglob_medium > 0) {
          medium_tags(igroup) = specfem::element::to_string(medium_tag);
          igroup++;

          const auto &field = forward.get_field<medium_tag>();

          typename OutputLibrary::Group group =
              base_group.createGroup(specfem::element::to_string(medium_tag));

          // Get the elements of the medium and their total
          const auto element_indices =
              element_types.get_elements_on_host(medium_tag);
          const int n_elements = element_indices.size();

          // Initialize the views
          DomainView x("xcoordinates", nglob_medium);
          DomainView z("zcoordinates", nglob_medium);
          MappingView mapping("mapping", n_elements, ngllz, ngllx);

          int ispec = 0;

          // Loop over the elements of the medium
          for (int iel = 0; iel < n_elements; iel++) {

            // Get the global element index
            ispec = element_indices(iel);

            for (int iz = 0; iz < ngllz; iz++) {
              for (int ix = 0; ix < ngllx; ix++) {

                // This is the local medium iglob
                // see: ``count`` in specfem::assembly::simulation_field<dim2,
                // medium>
                const int iglob = forward.template get_iglob<false, medium_tag>(
                    ispec, iz, ix);

                // Set the mapping for the medium element
                mapping(iel, iz, ix) = iglob;

                // Assign the coordinates to the local iglob
                x(iglob) = mesh.h_coord(0, ispec, iz, ix);
                z(iglob) = mesh.h_coord(1, ispec, iz, ix);
              }
            }
          }

          group.createDataset("X", x).write();
          group.createDataset("Z", z).write();
          group.createDataset("mapping", mapping).write();
        }
      }
    );

  file.createDataset("medium_tags", medium_tags).write();
  file.flush();

  specfem::Logger::info("Coordinates written to " + file_path);
}

template <typename OutputLibrary>
void specfem::io::wavefield_writer<OutputLibrary>::run(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly,
    const int istep) {
  auto &forward = assembly.fields.forward;

  forward.copy_to_host();

  typename OutputLibrary::Group base_group = file.createGroup(
      std::string("/Step") + specfem::utilities::to_zero_lead(istep, 6));

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) *
          MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                     elastic_psv_t),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;

        // Get the number of GLL points in the medium
        int nglob_medium = forward.get_nglob<medium_tag>();

        if (nglob_medium > 0) {
          const auto &field = forward.get_field<medium_tag>();

          typename OutputLibrary::Group group =
              base_group.createGroup(specfem::element::to_string(medium_tag));

          if constexpr (medium_tag == specfem::element::medium_tag::acoustic) {
            group.createDataset("Potential", field.get_host_field()).write();
            group.createDataset("PotentialDot", field.get_host_field_dot())
                .write();
            group
                .createDataset("PotentialDotDot",
                               field.get_host_field_dot_dot())
                .write();
          } else {
            group.createDataset("Displacement", field.get_host_field()).write();
            group.createDataset("Velocity", field.get_host_field_dot()).write();
            group.createDataset("Acceleration", field.get_host_field_dot_dot())
                .write();
          }
        }
      });

  file.flush();
}

template <typename OutputLibrary>
void specfem::io::wavefield_writer<OutputLibrary>::finalize(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly) {

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

    specfem::tag_dispatch::for_each(
        DIMENSION_SET(dim2) *
            MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                       elastic_psv_t),
        [&]<typename TagsType>() {
          constexpr auto medium_tag = TagsType::medium_tag;
          auto &ctr =
              boundary_values.stacey.container.template get<TagsType>();
          if (ctr.h_values.size() > 0) {
            const std::string dataset_name =
                specfem::element::to_string(medium_tag) + "Acceleration";
            stacey.createDataset(dataset_name, ctr.h_values).write();
          }
        });
    file.flush();
  }

  specfem::Logger::info("Wavefield written to " + file_path);
}
