#ifndef SPECFEM_WAVEFIELD_READER_TPP
#define SPECFEM_WAVEFIELD_READER_TPP

#include "reader/wavefield.hpp"
#include "IO/HDF5/HDF5.hpp"

template <typename IOLibrary>
specfem::reader::wavefield<IOLibrary>::wavefield(
    const std::string &output_folder,
    const specfem::compute::assembly &assembly)
    : output_folder(output_folder),
      elastic_field(assembly.fields.adjoint.elastic),
      acoustic_field(assembly.fields.adjoint.acoustic) {}

template <typename IOLibrary>
void specfem::reader::wavefield<IOLibrary>::read() {

  typename IOLibrary::File file(output_folder + "/ForwardWavefield");

  typename IOLibrary::Group elastic = file.openGroup("/Elastic");

  elastic.openDataset("Displacement", elastic_field.field).read();
  elastic.openDataset("Velocity", elastic_field.field_dot).read();
  elastic.openDataset("Acceleration", elastic_field.field_dot_dot).read();

  typename IOLibrary::Group acoustic = file.openGroup("/Acoustic");

  acoustic.openDataset("Potential", acoustic_field.field).read();
  acoustic.openDataset("PotentialDot", acoustic_field.field_dot).read();
  acoustic.openDataset("PotentialDotDot", acoustic_field.field_dot_dot).read();
}

#endif /* SPECFEM_WAVEFIELD_READER_TPP */
