#pragma once

#include "io/wavefield/writer.hpp"
#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "utilities/strings.hpp"

template <typename OutputLibrary>
specfem::io::wavefield_writer<OutputLibrary>::wavefield_writer(
    const std::string output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void specfem::io::wavefield_writer<OutputLibrary>::write(
    specfem::compute::assembly &assembly) {
  auto &forward = assembly.fields.forward;
  auto &boundary_values = assembly.boundary_values;

  forward.copy_to_host();
  boundary_values.copy_to_host();

  std::string dst = this->output_folder + "/ForwardWavefield";

  if (this->istep != -1) {
    dst += specfem::utilities::to_zero_lead(istep, 6);
  }

  typename OutputLibrary::File file(dst);

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC)),
      {
        typename OutputLibrary::Group group =
            file.createGroup(std::string("/") + specfem::element::to_string(_medium_tag_));
        const auto &field = forward.get_field<_medium_tag_>();

        if (_medium_tag_ == specfem::element::medium_tag::acoustic) {
          group.createDataset("Potential", field.h_field).write();
          group.createDataset("PotentialDot", field.h_field_dot).write();
          group.createDataset("PotentialDotDot", field.h_field_dot_dot).write();
        }
        else {
          group.createDataset("Displacement", field.h_field).write();
          group.createDataset("Velocity", field.h_field_dot).write();
          group.createDataset("Acceleration", field.h_field_dot_dot).write();
        }
      });

  typename OutputLibrary::Group boundary = file.createGroup("/Boundary");
  typename OutputLibrary::Group stacey = boundary.createGroup("/Stacey");

  stacey
      .createDataset("IndexMapping",
                     boundary_values.stacey.h_property_index_mapping)
      .write();
  stacey
      .createDataset("ElasticAcceleration",
                     boundary_values.stacey.elastic.h_values)
      .write();
  stacey
      .createDataset("AcousticAcceleration",
                     boundary_values.stacey.acoustic.h_values)
      .write();
  stacey
      .createDataset("PoroelasticAcceleration",
                     boundary_values.stacey.poroelastic.h_values)
      .write();

  std::cout << "Wavefield written to " << dst
            << std::endl;
}
