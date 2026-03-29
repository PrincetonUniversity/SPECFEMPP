#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/io/property/reader.hpp"

#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <typename InputLibrary>
specfem::io::property_reader<InputLibrary>::property_reader(
    const std::string &input_folder)
    : input_folder(input_folder) {}

template <typename InputLibrary>
void specfem::io::property_reader<InputLibrary>::read(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly) {
  auto &properties = assembly.properties;

  typename InputLibrary::File file(input_folder + "/Properties");

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T, ELASTIC_SPIN),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      {
        const std::string name =
            std::string("/") +
            specfem::element::to_string(_medium_tag_, _property_tag_);
        typename InputLibrary::Group group = file.openGroup(name);
        const auto container =
            properties.get_container<_medium_tag_, _property_tag_>();
        container.for_each_host_view(
            [&](const auto view, const std::string name) {
              group.openDataset(name, view).read();
            });
      })

  std::cout << "Properties read from " << input_folder << "/Properties"
            << std::endl;

  properties.copy_to_device();
}
