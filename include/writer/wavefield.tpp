#ifndef SPEC_WAVEFIELD_WRITER_TPP
#define SPEC_WAVEFIELD_WRITER_TPP

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "writer/wavefield.hpp"

template <typename OutputLibrary>
specfem::writer::wavefield<OutputLibrary>::wavefield(
    const specfem::compute::assembly &assembly, const std::string output_folder)
    : output_folder(output_folder), forward(assembly.fields.forward) {}

template <typename OutputLibrary>
void specfem::writer::wavefield<OutputLibrary>::write() {

  typename OutputLibrary::File file(output_folder + "/ForwardWavefield");

  typename OutputLibrary::Group elastic = file.createGroup("/Elastic");
  typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");

  elastic.createDataset("Displacement", forward.elastic.field).write();
  elastic.createDataset("Velocity", forward.elastic.field_dot).write();
  elastic.createDataset("Acceleration", forward.elastic.field_dot_dot).write();

  acoustic.createDataset("Potential", forward.acoustic.field).write();
  acoustic.createDataset("PotentialDot", forward.acoustic.field_dot).write();
  acoustic.createDataset("PotentialDotDot", forward.acoustic.field_dot_dot)
      .write();

  std::cout << "Wavefield written to " << output_folder + "/ForwardWavefield"
            << std::endl;
}

#endif /* SPEC_WAVEFIELD_WRITER_TPP */
