#include "parameter_parser/writer/kernel.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "writer/interface.hpp"
#include "writer/kernel.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::kernel::kernel(
    const YAML::Node &Node, const specfem::simulation::type type) {

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
    *this = specfem::runtime_configuration::kernel(
        Node["output-format"].as<std::string>(), output_folder, type);
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading wavefield config. \n" << e.what();

    std::runtime_error(message.str());
  }

  return;
}

std::shared_ptr<specfem::writer::writer>
specfem::runtime_configuration::kernel::instantiate_kernel_writer(
    const specfem::compute::assembly &assembly) const {

  const std::shared_ptr<specfem::writer::writer> writer =
      [&]() -> std::shared_ptr<specfem::writer::writer> {
    if (this->simulation_type == specfem::simulation::type::combined) {
      if (this->output_format == "HDF5") {
        return std::make_shared<
            specfem::writer::kernel<specfem::IO::HDF5<specfem::IO::write> > >(
            assembly, this->output_folder);
      } else if (this->output_format == "ASCII") {
        return std::make_shared<
            specfem::writer::kernel<specfem::IO::ASCII<specfem::IO::write> > >(
            assembly, this->output_folder);
      } else {
        throw std::runtime_error("Unknown wavefield format");
      }
    } else {
      return nullptr;
    }
  }();

  return writer;
}
