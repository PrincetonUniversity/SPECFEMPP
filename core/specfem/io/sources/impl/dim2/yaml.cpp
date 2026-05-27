#include "yaml-cpp/yaml.h"
#include "specfem/io/sources/impl/reader.hpp"
#include "specfem/source.hpp"

#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim2>>>
specfem::io::sources_impl::read<specfem::element::dimension_tag::dim2,
                                specfem::enums::source_format::YAML>(
    const YAML::Node &source_dict, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type) {

  // Extract source sequence from the sources node
  YAML::Node file_sources = source_dict["sources"];

  // Double check that the sources are indeed a list
  assert(file_sources.IsSequence());
  assert(file_sources.size() > 0);

  // Validate number-of-sources if present
  if (source_dict["number-of-sources"]) {
    int nsources = source_dict["number-of-sources"].as<int>();
    if (static_cast<int>(file_sources.size()) != nsources) {
      std::ostringstream message;
      message << "Found only " << file_sources.size()
              << " number of sources. Found total number of sources in are "
              << nsources
              << " Please check if there is a error in sources file.";
      throw std::runtime_error(message.str());
    }
  }

  std::vector<std::shared_ptr<
      specfem::sources::source<specfem::element::dimension_tag::dim2>>>
      sources;

  // Note: Make sure you name the YAML node different from the name of the
  // source class Otherwise, the compiler will get confused and throw an error
  // I've spent hours debugging this issue. It is very annoying since it only
  // shows up on CUDA compiler
  for (auto N : file_sources) {
    if (YAML::Node force_source = N["force"]) {
      sources.push_back(
          std::make_shared<
              specfem::sources::force<specfem::element::dimension_tag::dim2>>(
              force_source, nsteps, dt, wavefield_type));
    } else if (YAML::Node cosserat_force = N["cosserat-force"]) {
      sources.push_back(std::make_shared<specfem::sources::cosserat_force<
                            specfem::element::dimension_tag::dim2>>(
          cosserat_force, nsteps, dt, wavefield_type));
    } else if (YAML::Node moment_tensor_source = N["moment-tensor"]) {
      sources.push_back(std::make_shared<specfem::sources::moment_tensor<
                            specfem::element::dimension_tag::dim2>>(
          moment_tensor_source, nsteps, dt, wavefield_type));
    } else if (YAML::Node external_source = N["user-defined"]) {
      sources.push_back(std::make_shared<specfem::sources::external<
                            specfem::element::dimension_tag::dim2>>(
          external_source, nsteps, dt, wavefield_type));
    } else if (YAML::Node adjoint_node = N["adjoint-source"]) {
      if (!adjoint_node["station_name"] || !adjoint_node["network_name"]) {
        throw std::runtime_error(
            "Station name and network name are required for adjoint source");
      }
      sources.push_back(std::make_shared<specfem::sources::adjoint_source<
                            specfem::element::dimension_tag::dim2>>(
          adjoint_node, nsteps, dt));
    } else {
      throw std::runtime_error("Unknown source type in YAML source file");
    }
  }

  return sources;
}

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim2>>>
specfem::io::sources_impl::read<specfem::element::dimension_tag::dim2,
                                specfem::enums::source_format::YAML>(
    const std::string &file_path, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type) {
  YAML::Node source_dict = YAML::LoadFile(file_path);
  return read<specfem::element::dimension_tag::dim2,
              specfem::enums::source_format::YAML>(source_dict, nsteps, dt,
                                                   wavefield_type);
}
