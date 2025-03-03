#include "parameter_parser/writer/kernel.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/kernel/writer.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::kernel::kernel(
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

  *this = specfem::runtime_configuration::kernel(output_format, output_folder,
                                                 type);

  return;
}

std::shared_ptr<specfem::IO::writer>
specfem::runtime_configuration::kernel::instantiate_kernel_writer() const {

  const std::shared_ptr<specfem::IO::writer> writer =
      [&]() -> std::shared_ptr<specfem::IO::writer> {
    if (this->simulation_type == specfem::simulation::type::combined) {
      if (this->output_format == "HDF5") {
        return std::make_shared<specfem::IO::kernel_writer<
            specfem::IO::HDF5<specfem::IO::write> > >(this->output_folder);
      } else if (this->output_format == "ASCII") {
        return std::make_shared<specfem::IO::kernel_writer<
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
