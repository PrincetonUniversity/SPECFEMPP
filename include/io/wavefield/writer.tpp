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

  typename OutputLibrary::Group elastic_psv = file.createGroup("/ElasticPSV");
  const auto &elastic_psv_field =
      forward.get_field<specfem::element::medium_tag::elastic_psv>();
  typename OutputLibrary::Group elastic_sh = file.createGroup("/ElasticSH");
  const auto &elastic_sh_field =
      forward.get_field<specfem::element::medium_tag::elastic_sh>();
  typename OutputLibrary::Group elastic_psv_t = file.createGroup("/ElasticPSVT");
  const auto &elastic_psv_t_field =
      forward.get_field<specfem::element::medium_tag::elastic_psv_t>();
  typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");
  const auto &acoustic_field =
      forward.get_field<specfem::element::medium_tag::acoustic>();
  typename OutputLibrary::Group poroelastic = file.createGroup("/Poroelastic");
  const auto &poroelastic_field =
      forward.get_field<specfem::element::medium_tag::poroelastic>();
  typename OutputLibrary::Group boundary = file.createGroup("/Boundary");
  typename OutputLibrary::Group stacey = boundary.createGroup("/Stacey");

  elastic_psv.createDataset("Displacement", elastic_psv_field.h_field).write();
  elastic_psv.createDataset("Velocity", elastic_psv_field.h_field_dot).write();
  elastic_psv.createDataset("Acceleration", elastic_psv_field.h_field_dot_dot)
      .write();

  elastic_sh.createDataset("Displacement", elastic_sh_field.h_field).write();
  elastic_sh.createDataset("Velocity", elastic_sh_field.h_field_dot).write();
  elastic_sh.createDataset("Acceleration", elastic_sh_field.h_field_dot_dot)
      .write();

  elastic_psv_t.createDataset("Displacement", elastic_psv_t_field.h_field)
      .write();
  elastic_psv_t.createDataset("Velocity", elastic_psv_t_field.h_field_dot)
      .write();
  elastic_psv_t.createDataset("Acceleration", elastic_psv_t_field.h_field_dot_dot)
      .write();

  acoustic.createDataset("Potential", acoustic_field.h_field).write();
  acoustic.createDataset("PotentialDot", acoustic_field.h_field_dot).write();
  acoustic.createDataset("PotentialDotDot", acoustic_field.h_field_dot_dot).write();

  poroelastic.createDataset("Displacement", poroelastic_field.h_field).write();
  poroelastic.createDataset("Velocity", poroelastic_field.h_field_dot).write();
  poroelastic.createDataset("Acceleration", poroelastic_field.h_field_dot_dot)
      .write();

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
