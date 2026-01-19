#pragma once

#include "io/wavefield/reader.hpp"
#include "specfem/utilities.hpp"

template <typename IOLibrary>
specfem::io::wavefield_reader<IOLibrary>::wavefield_reader(
    const std::string &output_folder)
    : output_folder(output_folder),
      file(typename IOLibrary::File(output_folder + "/ForwardWavefield")) {}

template <typename IOLibrary>
void specfem::io::wavefield_reader<IOLibrary>::initialize(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  auto &buffer = assembly.fields.buffer;
  int ngroups = 0;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_PSV_T, ELASTIC_SH,
                                       ACOUSTIC, POROELASTIC)),
      {
        if (buffer.get_nglob<_medium_tag_>() > 0) {
          ngroups++;
        }
      });

  Kokkos::View<std::string *, Kokkos::HostSpace> medium_tags("medium_tags", ngroups);
  file.openDataset("medium_tags", medium_tags).read();

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_PSV_T, ELASTIC_SH,
                                       ACOUSTIC, POROELASTIC)),
      {
        if (buffer.get_nglob<_medium_tag_>() > 0) {
          const std::string current_tag = specfem::element::to_string(_medium_tag_);
          bool found = false;
          for (int i = 0; i < medium_tags.extent(0); ++i) {
            if (current_tag == medium_tags(i)) {
              found = true;
              break;
            }
          }
          if (!found) {
            throw std::runtime_error("Medium tag " + specfem::element::to_string(_medium_tag_) + " not found in wavefield file");
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

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE((container, boundary_values.stacey.container)) {
        if (_container_.h_values.size() > 0) {
          const std::string dataset_name =
              specfem::element::to_string(_medium_tag_) + "Acceleration";
          stacey.openDataset(dataset_name, _container_.h_values).read();
        }
      });

  boundary_values.copy_to_device();

  return;
}

template <typename IOLibrary>
void specfem::io::wavefield_reader<IOLibrary>::run(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const int istep) {
  auto &buffer = assembly.fields.buffer;
  auto &boundary_values = assembly.boundary_values;

  typename IOLibrary::Group base_group = file.openGroup(
      std::string("/Step") + specfem::utilities::to_zero_lead(istep, 6));

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        int nglob_medium = buffer.get_nglob<_medium_tag_>();

        if (nglob_medium > 0) {
          typename IOLibrary::Group group =
              base_group.openGroup(specfem::element::to_string(_medium_tag_));
          const auto &field = buffer.get_field<_medium_tag_>();

          if (_medium_tag_ == specfem::element::medium_tag::acoustic) {
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
