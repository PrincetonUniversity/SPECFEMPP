#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "io/property/reader.hpp"
#include "kokkos_abstractions.h"
#include "point/properties.hpp"
#include <Kokkos_Core.hpp>

template <typename InputLibrary>
specfem::io::property_reader<InputLibrary>::property_reader(
    const std::string input_folder)
    : input_folder(input_folder) {}

template <typename InputLibrary>
void specfem::io::property_reader<InputLibrary>::read(
    specfem::compute::assembly &assembly) {
  auto &properties = assembly.properties;

  typename InputLibrary::File file(input_folder + "/Properties");

  FOR_EACH_MATERIAL_SYSTEM(
      IN((DIMENSION_TAG_DIM2),
         (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC),
         (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
      {
        const std::string name =
            std::string("/") +
            specfem::element::to_string(_medium_tag_, _property_tag_);
        typename InputLibrary::Group group = file.openGroup(name);
        const auto container =
            properties.get_container<_medium_tag_, _property_tag_>();
        container.for_each_host_view([&](const auto view, const std::string name) {
          group.openDataset(name, view).read();
        });
      })

  std::cout << "Properties read from " << input_folder << "/Properties"
            << std::endl;

  properties.copy_to_device();
}
