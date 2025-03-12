#pragma once

#include "IO/property/reader.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/properties.hpp"
#include <Kokkos_Core.hpp>

template <typename InputLibrary>
specfem::IO::property_reader<InputLibrary>::property_reader(
    const std::string input_folder)
    : input_folder(input_folder) {}

template <typename InputLibrary>
void specfem::IO::property_reader<InputLibrary>::read(
    specfem::compute::assembly &assembly) {
  auto &properties = assembly.properties;

  typename InputLibrary::File file(input_folder + "/Properties");

#define READ_PROPERTY(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)                 \
  {                                                                            \
    const std::string name =                                                   \
        std::string("/") + specfem::element::to_string(GET_TAG(MEDIUM_TAG),    \
                                                       GET_TAG(PROPERTY_TAG)); \
    typename InputLibrary::Group group = file.openGroup(name);                 \
    group                                                                      \
        .openDataset(                                                          \
            "data",                                                            \
            properties                                                         \
                .get_container<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>()   \
                .h_data)                                                       \
        .read();                                                               \
  }

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      READ_PROPERTY,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC));

#undef READ_PROPERTY

  // {
  //   typename InputLibrary::Group elastic =
  //       file.openGroup("/ElasticSVIsotropic");

  //   elastic
  //       .openDataset(
  //           "data",
  //           properties
  //               .get_container<specfem::element::medium_tag::elastic_sv,
  //                              specfem::element::property_tag::isotropic>()
  //               .h_data)
  //       .read();
  // }

  // {
  //   typename InputLibrary::Group elastic =
  //       file.openGroup("/ElasticSHIsotropic");

  //   elastic
  //       .openDataset(
  //           "data",
  //           properties
  //               .get_container<specfem::element::medium_tag::elastic_sh,
  //                              specfem::element::property_tag::isotropic>()
  //               .h_data)
  //       .read();
  // }

  // {
  //   typename InputLibrary::Group elastic =
  //       file.openGroup("/ElasticSVAnisotropic");

  //   elastic
  //       .openDataset(
  //           "data",
  //           properties
  //               .get_container<specfem::element::medium_tag::elastic_sv,
  //                              specfem::element::property_tag::anisotropic>()
  //               .h_data)
  //       .read();
  // }

  // {
  //   typename InputLibrary::Group elastic =
  //       file.openGroup("/ElasticSHAnisotropic");

  //   elastic
  //       .openDataset(
  //           "data",
  //           properties
  //               .get_container<specfem::element::medium_tag::elastic_sh,
  //                              specfem::element::property_tag::anisotropic>()
  //               .h_data)
  //       .read();
  // }

  // {
  //   typename InputLibrary::Group acoustic = file.openGroup("/Acoustic");

  //   acoustic
  //       .openDataset(
  //           "data",
  //           properties
  //               .get_container<specfem::element::medium_tag::acoustic,
  //                              specfem::element::property_tag::isotropic>()
  //               .h_data)
  //       .read();
  // }

  std::cout << "Properties read from " << input_folder << "/Properties"
            << std::endl;

  properties.copy_to_device();
}
