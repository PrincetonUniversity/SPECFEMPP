#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/io/property/reader.hpp"

#include "specfem/mpi.hpp"
#include "specfem/point.hpp"
#include <boost/filesystem.hpp>
#include <Kokkos_Core.hpp>

template <typename InputLibrary>
specfem::io::property_reader<InputLibrary>::property_reader(
    const std::string &input_folder)
    : input_folder(input_folder) {}

template <typename InputLibrary>
void specfem::io::property_reader<InputLibrary>::read(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly) {
  auto &properties = assembly.properties;

  // Build rank-specific input path following the same convention as mesh files:
  //   serial:   {input_folder}/Properties/
  //   parallel: {input_folder}/Properties/proc_N/
  const std::string formatted =
      specfem::MPI::format_proc_filename(input_folder + "/Properties.dir");
  const boost::filesystem::path formatted_path(formatted);
  const std::string base_folder = formatted_path.parent_path().string();
  const std::string ns = formatted_path.stem().string();

  typename InputLibrary::File file(base_folder + "/" + ns);

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
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

  std::cout << "Properties read from " << base_folder << "/" << ns
            << std::endl;

  properties.copy_to_device();
}
