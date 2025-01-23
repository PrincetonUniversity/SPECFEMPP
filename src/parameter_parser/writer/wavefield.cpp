#include "parameter_parser/writer/wavefield.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/reader.hpp"
#include "IO/wavefield/reader.hpp"
#include "IO/wavefield/writer.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::wavefield::wavefield(
    const YAML::Node &Node, const specfem::simulation::type type) {

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

  *this = specfem::runtime_configuration::wavefield(output_format,
                                                    output_folder, type);

  return;
}

std::shared_ptr<specfem::IO::writer>
specfem::runtime_configuration::wavefield::instantiate_wavefield_writer()
    const {

  const std::shared_ptr<specfem::IO::writer> writer =
      [&]() -> std::shared_ptr<specfem::IO::writer> {
    if (this->simulation_type == specfem::simulation::type::forward) {
      if (this->output_format == "HDF5") {
        return std::make_shared<specfem::IO::wavefield_writer<
            specfem::IO::HDF5<specfem::IO::write> > >(this->output_folder);
      } else if (this->output_format == "ASCII") {
        return std::make_shared<specfem::IO::wavefield_writer<
            specfem::IO::ASCII<specfem::IO::write> > >(this->output_folder);
      } else {
        throw std::runtime_error("Unknown wavefield format");
      }
    } else {
      return nullptr;
    }
  }();

  return writer;
}

std::shared_ptr<specfem::IO::reader>
specfem::runtime_configuration::wavefield::instantiate_wavefield_reader()
    const {

  const std::shared_ptr<specfem::IO::reader> reader =
      [&]() -> std::shared_ptr<specfem::IO::reader> {
    if (this->simulation_type == specfem::simulation::type::combined) {
      if (this->output_format == "HDF5") {
        return std::make_shared<specfem::IO::wavefield_reader<
            specfem::IO::HDF5<specfem::IO::read> > >(this->output_folder);
      } else if (this->output_format == "ASCII") {
        return std::make_shared<specfem::IO::wavefield_reader<
            specfem::IO::ASCII<specfem::IO::read> > >(this->output_folder);
      } else {
        throw std::runtime_error("Unknown wavefield format");
      }
    } else {
      return nullptr;
    }
  }();

  return reader;
}
