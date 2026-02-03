#include "kernel.hpp"
#include "io/ADIOS2/ADIOS2.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/NPY/NPY.hpp"
#include "io/NPZ/NPZ.hpp"
#include "io/kernel/writer.hpp"
#include "specfem/utilities.hpp"
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

std::shared_ptr<specfem::io::writer>
specfem::runtime_configuration::kernel::instantiate_kernel_writer() const {

  const std::shared_ptr<specfem::io::writer> writer =
      [&]() -> std::shared_ptr<specfem::io::writer> {
    if (this->simulation_type == specfem::simulation::type::combined) {
      if (specfem::utilities::is_hdf5_string(this->output_format)) {
        return std::make_shared<specfem::io::kernel_writer<
            specfem::io::HDF5<specfem::io::write> > >(this->output_folder);
      } else if (specfem::utilities::is_adios2_string(this->output_format)) {
        return std::make_shared<specfem::io::kernel_writer<
            specfem::io::ADIOS2<specfem::io::write> > >(this->output_folder);
      } else if (specfem::utilities::is_ascii_string(this->output_format)) {
        return std::make_shared<specfem::io::kernel_writer<
            specfem::io::ASCII<specfem::io::write> > >(this->output_folder);
      } else if (specfem::utilities::is_npy_string(this->output_format)) {
        return std::make_shared<
            specfem::io::kernel_writer<specfem::io::NPY<specfem::io::write> > >(
            this->output_folder);
      } else if (specfem::utilities::is_npz_string(this->output_format)) {
        return std::make_shared<
            specfem::io::kernel_writer<specfem::io::NPZ<specfem::io::write> > >(
            this->output_folder);
      } else {
        throw std::runtime_error("Unknown wavefield format");
      }
    } else {
      return nullptr;
    }
  }();

  return writer;
}
