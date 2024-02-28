#include "parameter_parser/wavefield.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "writer/interface.hpp"
#include "writer/wavefield.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::wavefield::wavefield(const YAML::Node &Node) {

  boost::filesystem::path cwd = boost::filesystem::current_path();
  std::string output_folder = cwd.string();
  if (Node["output-folder"]) {
    output_folder = Node["output-folder"].as<std::string>();
  }

  if (!boost::filesystem::is_directory(
          boost::filesystem::path(output_folder))) {
    std::ostringstream message;
    message << "Output folder : " << output_folder << " does not exist.";
    throw std::runtime_error(message.str());
  }

  try {
    *this = specfem::runtime_configuration::wavefield(
        Node["wavefield-format"].as<std::string>(), output_folder);
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading wavefield config. \n" << e.what();

    std::runtime_error(message.str());
  }

  return;
}

std::shared_ptr<specfem::writer::writer>
specfem::runtime_configuration::wavefield::instantiate_wavefield_writer(
    const specfem::compute::assembly &assembly) const {

  const std::shared_ptr<specfem::writer::writer> writer =
      [&]() -> std::shared_ptr<specfem::writer::writer> {
    if (this->wavefield_format == "HDF5") {
      return std::make_shared<
          specfem::writer::wavefield<specfem::IO::HDF5<specfem::IO::write> > >(
          assembly, this->output_folder);
    } else {
      throw std::runtime_error("Unknown wavefield format");
    }
  }();

  return writer;
}

std::shared_ptr<specfem::reader::reader>
specfem::runtime_configuration::wavefield::instantiate_wavefield_reader(
    const specfem::compute::assembly &assembly) const {

  const std::shared_ptr<specfem::reader::reader> reader =
      [&]() -> std::shared_ptr<specfem::reader::reader> {
    if (this->wavefield_format == "HDF5") {
      return std::make_shared<
          specfem::reader::wavefield<specfem::IO::HDF5<specfem::IO::read> > >(
          this->output_folder, assembly);
    } else {
      throw std::runtime_error("Unknown wavefield format");
    }
  }();

  return reader;
}
