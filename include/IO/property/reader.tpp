#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/properties.hpp"
#include "IO/property/reader.hpp"
#include <Kokkos_Core.hpp>

template <typename InputLibrary>
specfem::IO::property_reader<InputLibrary>::property_reader(const std::string input_folder): input_folder(input_folder) {}

template <typename InputLibrary>
void specfem::IO::property_reader<InputLibrary>::read(specfem::compute::assembly &assembly) {
  auto &properties = assembly.properties;

  using DomainView =
      Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  typename InputLibrary::File file(input_folder + "/Properties");

  {
    typename InputLibrary::Group elastic = file.openGroup("/ElasticIsotropic");

    elastic.openDataset("data", properties.elastic_isotropic.h_data).read();
  }

  {
    typename InputLibrary::Group elastic = file.openGroup("/ElasticAnisotropic");

    elastic.openDataset("data", properties.elastic_anisotropic.h_data).read();
  }

  {
    typename InputLibrary::Group acoustic = file.openGroup("/Acoustic");

    acoustic.openDataset("data", properties.acoustic_isotropic.h_data).read();
  }

  std::cout << "Properties read from " << input_folder << "/Properties"
            << std::endl;

  properties.copy_to_device();
}
