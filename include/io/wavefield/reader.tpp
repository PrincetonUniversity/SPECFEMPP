#pragma once

#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/wavefield/reader.hpp"
#include "utilities/strings.hpp"

template <typename IOLibrary>
specfem::io::wavefield_reader<IOLibrary>::wavefield_reader(
    const std::string &output_folder)
    : output_folder(output_folder) {}

template <typename IOLibrary>
void specfem::io::wavefield_reader<IOLibrary>::read(
    specfem::compute::assembly &assembly) {
  auto &buffer = assembly.fields.buffer;
  auto &boundary_values = assembly.boundary_values;

  std::string dst = this->output_folder + "/ForwardWavefield";

  if (this->istep != -1) {
    dst += specfem::utilities::to_zero_lead(istep, 6);
  }

  typename IOLibrary::File file(dst);

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC)),
      {
        typename OutputLibrary::Group group =
            file.createGroup(std::string("/") + specfem::element::to_string(_medium_tag_));
        const auto &field = forward.get_field<_medium_tag_>();

        if (_medium_tag_ == specfem::element::medium_tag::acoustic) {
          group.openDataset("Potential", field.h_field).read();
          group.openDataset("PotentialDot", field.h_field_dot).read();
          group.openDataset("PotentialDotDot", field.h_field_dot_dot).read();
        }
        else {
          group.openDataset("Displacement", field.h_field).read();
          group.openDataset("Velocity", field.h_field_dot).read();
          group.openDataset("Acceleration", field.h_field_dot_dot).read();
        }
      });

  typename IOLibrary::Group boundary = file.openGroup("/Boundary");
  typename IOLibrary::Group stacey = boundary.openGroup("/Stacey");

  stacey
      .openDataset("IndexMapping",
                   boundary_values.stacey.h_property_index_mapping)
      .read();
  stacey
      .openDataset("ElasticAcceleration",
                   boundary_values.stacey.elastic.h_values)
      .read();
  stacey
      .openDataset("AcousticAcceleration",
                   boundary_values.stacey.acoustic.h_values)
      .read();
  stacey
      .openDataset("PoroelasticAcceleration",
                   boundary_values.stacey.poroelastic.h_values)
      .read();

  buffer.copy_to_device();
  boundary_values.copy_to_device();
}
