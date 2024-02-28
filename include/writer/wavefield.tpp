#ifndef SPEC_WAVEFIELD_WRITER_TPP
#define SPEC_WAVEFIELD_WRITER_TPP

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "writer/wavefield.hpp"

template <typename OutputLibrary>
specfem::writer::wavefield<OutputLibrary>::wavefield(
    const specfem::compute::assembly &assembly, const std::string output_folder)
    : output_folder(output_folder),
      elastic_field(assembly.fields.forward.elastic),
      acoustic_field(assembly.fields.forward.acoustic) {}

template <typename OutputLibrary>
void specfem::writer::wavefield<OutputLibrary>::write() {

  typename OutputLibrary::File file(output_folder + "/ForwardWavefield");

  typename OutputLibrary::Group elastic = file.createGroup("/Elastic");
  typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");

  elastic.createDataset("Displacement", elastic_field.field).write();
  elastic.createDataset("Velocity", elastic_field.field_dot).write();
  elastic.createDataset("Acceleration", elastic_field.field_dot_dot).write();

  acoustic.createDataset("Potential", acoustic_field.field).write();
  acoustic.createDataset("PotentialDot", acoustic_field.field_dot).write();
  acoustic.createDataset("PotentialDotDot", acoustic_field.field_dot_dot)
      .write();
}

#endif /* SPEC_WAVEFIELD_WRITER_TPP */
