#include "parameter_parser/writer/property.hpp"
#include "io/ADIOS2/ADIOS2.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/NPY/NPY.hpp"
#include "io/NPZ/NPZ.hpp"
#include "io/property/reader.hpp"
#include "io/property/writer.hpp"
#include "utilities/strings.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::property::property(const YAML::Node &Node,
                                                   const bool write_mode) {

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
    message << "Model folder : " << output_folder << " does not exist.";
    throw std::runtime_error(message.str());
  }

  *this = specfem::runtime_configuration::property(output_format, output_folder,
                                                   write_mode);

  return;
}

std::shared_ptr<specfem::io::writer>
specfem::runtime_configuration::property::instantiate_property_writer() const {

  const std::shared_ptr<specfem::io::writer> writer =
      [&]() -> std::shared_ptr<specfem::io::writer> {
    if (!this->write_mode) {
      return nullptr;
    }
    if (specfem::utilities::is_hdf5_string(this->output_format)) {
      return std::make_shared<specfem::io::property_writer<
          specfem::io::HDF5<specfem::io::write> > >(this->output_folder);
    } else if (specfem::utilities::is_adios2_string(this->output_format)) {
      return std::make_shared<specfem::io::property_writer<
          specfem::io::ADIOS2<specfem::io::write> > >(this->output_folder);
    } else if (specfem::utilities::is_ascii_string(this->output_format)) {
      return std::make_shared<specfem::io::property_writer<
          specfem::io::ASCII<specfem::io::write> > >(this->output_folder);
    } else if (specfem::utilities::is_npy_string(this->output_format)) {
      return std::make_shared<
          specfem::io::property_writer<specfem::io::NPY<specfem::io::write> > >(
          this->output_folder);
    } else if (specfem::utilities::is_npz_string(this->output_format)) {
      return std::make_shared<
          specfem::io::property_writer<specfem::io::NPZ<specfem::io::write> > >(
          this->output_folder);
    } else {
      throw std::runtime_error("Unknown model format");
    }
  }();

  return writer;
}

std::shared_ptr<specfem::io::reader>
specfem::runtime_configuration::property::instantiate_property_reader() const {

  const std::shared_ptr<specfem::io::reader> reader =
      [&]() -> std::shared_ptr<specfem::io::reader> {
    if (this->write_mode) {
      return nullptr;
    }
    if (specfem::utilities::is_hdf5_string(this->output_format)) {
      return std::make_shared<
          specfem::io::property_reader<specfem::io::HDF5<specfem::io::read> > >(
          this->output_folder);
    } else if (specfem::utilities::is_adios2_string(this->output_format)) {
      return std::make_shared<specfem::io::property_reader<
          specfem::io::ADIOS2<specfem::io::read> > >(this->output_folder);
    } else if (specfem::utilities::is_ascii_string(this->output_format)) {
      return std::make_shared<specfem::io::property_reader<
          specfem::io::ASCII<specfem::io::read> > >(this->output_folder);
    } else if (specfem::utilities::is_npy_string(this->output_format)) {
      return std::make_shared<
          specfem::io::property_reader<specfem::io::NPY<specfem::io::read> > >(
          this->output_folder);
    } else if (specfem::utilities::is_npz_string(this->output_format)) {
      return std::make_shared<
          specfem::io::property_reader<specfem::io::NPZ<specfem::io::read> > >(
          this->output_folder);
    } else {
      throw std::runtime_error("Unknown model format");
    }
  }();

  return reader;
}
