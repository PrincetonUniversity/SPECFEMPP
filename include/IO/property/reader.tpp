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

    elastic.openDataset("rho", properties.elastic_isotropic.h_rho).read();
    elastic.openDataset("mu", properties.elastic_isotropic.h_mu).read();
    elastic.openDataset("lambdaplus2mu", properties.elastic_isotropic.h_lambdaplus2mu).read();
  }

  {
    typename InputLibrary::Group elastic = file.openGroup("/ElasticAnisotropic");

    elastic.openDataset("rho", properties.elastic_anisotropic.h_rho).read();
    elastic.openDataset("c11", properties.elastic_anisotropic.h_c11).read();
    elastic.openDataset("c13", properties.elastic_anisotropic.h_c13).read();
    elastic.openDataset("c15", properties.elastic_anisotropic.h_c15).read();
    elastic.openDataset("c33", properties.elastic_anisotropic.h_c33).read();
    elastic.openDataset("c35", properties.elastic_anisotropic.h_c35).read();
    elastic.openDataset("c55", properties.elastic_anisotropic.h_c55).read();
    elastic.openDataset("c12", properties.elastic_anisotropic.h_c12).read();
    elastic.openDataset("c23", properties.elastic_anisotropic.h_c23).read();
    elastic.openDataset("c25", properties.elastic_anisotropic.h_c25).read();
  }

  {
    typename InputLibrary::Group acoustic = file.openGroup("/Acoustic");

    acoustic.openDataset("rho_inverse", properties.acoustic_isotropic.h_rho_inverse).read();
    acoustic.openDataset("kappa", properties.acoustic_isotropic.h_kappa).read();
  }

  std::cout << "Properties read from " << input_folder << "/Properties"
            << std::endl;

  properties.copy_to_device();
}
