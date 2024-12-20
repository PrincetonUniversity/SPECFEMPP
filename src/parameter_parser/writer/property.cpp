#include "parameter_parser/writer/property.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "writer/interface.hpp"
#include "writer/property.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::property::property(
    const YAML::Node &Node) {

  const std::string output_format = [&]() -> std::string {
    if (Node["format"]) {
      return Node["format"].as<std::string>();
    } else {
      return "ASCII";
    }
  }();

  const std::string output_folder = [&]() -> std::string {
    if (Node["directory"]) {
      return Node["directory"].as<std::string>();
    } else {
      return boost::filesystem::current_path().string();
    }
  }();

  if (!boost::filesystem::is_directory(
          boost::filesystem::path(output_folder))) {
    std::ostringstream message;
    message << "Output folder : " << output_folder << " does not exist.";
    throw std::runtime_error(message.str());
  }

  *this = specfem::runtime_configuration::property(output_format, output_folder);

  return;
}

std::shared_ptr<specfem::writer::writer>
specfem::runtime_configuration::property::instantiate_property_writer(
    const specfem::compute::assembly &assembly) const {

  const std::shared_ptr<specfem::writer::writer> writer =
      [&]() -> std::shared_ptr<specfem::writer::writer> {
      if (this->output_format == "HDF5") {
        return std::make_shared<
            specfem::writer::property<specfem::IO::HDF5<specfem::IO::write> > >(
            assembly, this->output_folder);
      } else if (this->output_format == "ASCII") {
        return std::make_shared<
            specfem::writer::property<specfem::IO::ASCII<specfem::IO::write> > >(
            assembly, this->output_folder);
      } else {
        throw std::runtime_error("Unknown model format");
      }
  }();

  return writer;
}
