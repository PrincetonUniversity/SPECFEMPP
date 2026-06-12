#include "wavefield.hpp"
#include "specfem/io.hpp"
#include "specfem/periodic_tasks.hpp"
#include "specfem/utilities.hpp"
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

  const int time_interval = [&]() -> int {
    if (Node["time-interval"]) {
      return Node["time-interval"].as<int>();
    } else {
      return 0;
    }
  }();

  const std::string time_interval_by_memory = [&]() -> std::string {
    if (Node["time-interval-by-memory"]) {
      if (time_interval != 0) {
        throw std::runtime_error(
            "time-interval and time-interval-by-memory cannot be used "
            "simultaneously");
      }
      return Node["time-interval-by-memory"].as<std::string>();
    } else {
      return "";
    }
  }();

  const bool include_last_step = [&]() -> bool {
    if (Node["include-last-step"]) {
      return Node["include-last-step"].as<bool>();
    } else {
      return true;
    }
  }();

  const bool for_adjoint_simulations = [&]() -> bool {
    if (Node["for-adjoint-simulations"]) {
      return Node["for-adjoint-simulations"].as<bool>();
    } else {
      return false;
    }
  }();

  if (time_interval == 0 && !include_last_step) {
    std::ostringstream message;
    message << "************************************************\n"
            << "Warning : Wavefield writer does not write any wavefield. \n"
            << "************************************************\n";
    std::cout << message.str();
  }

  *this = specfem::runtime_configuration::wavefield(
      output_format, output_folder, type, time_interval,
      time_interval_by_memory, include_last_step, for_adjoint_simulations);

  return;
}

template <specfem::element::dimension_tag DimensionTag>
std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag>>
specfem::runtime_configuration::wavefield::instantiate_wavefield_writer()
    const {

  const std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag>>
      writer = [&]()
      -> std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag>> {
    if (this->simulation_type == specfem::simulation::type::forward) {
      if (specfem::utilities::is_hdf5_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_writer<
            DimensionTag, specfem::io_backends::HDF5>>(
            this->output_folder, this->time_interval, this->include_last_step,
            this->for_adjoint_simulations);
      } else if (specfem::utilities::is_adios2_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_writer<
            DimensionTag, specfem::io_backends::ADIOS2>>(
            this->output_folder, this->time_interval, this->include_last_step,
            this->for_adjoint_simulations);
      } else if (specfem::utilities::is_ascii_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_writer<
            DimensionTag, specfem::io_backends::ASCII>>(
            this->output_folder, this->time_interval, this->include_last_step,
            this->for_adjoint_simulations);
      } else if (specfem::utilities::is_npy_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_writer<
            DimensionTag, specfem::io_backends::NPY>>(
            this->output_folder, this->time_interval, this->include_last_step,
            this->for_adjoint_simulations);
      } else if (specfem::utilities::is_npz_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_writer<
            DimensionTag, specfem::io_backends::NPZ>>(
            this->output_folder, this->time_interval, this->include_last_step,
            this->for_adjoint_simulations);
      } else {
        throw std::runtime_error("Unknown wavefield format");
      }
    } else {
      return nullptr;
    }
  }();

  return writer;
}

template <specfem::element::dimension_tag DimensionTag>
std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag>>
specfem::runtime_configuration::wavefield::instantiate_wavefield_reader()
    const {

  const std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag>>
      reader = [&]()
      -> std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag>> {
    if (this->simulation_type == specfem::simulation::type::combined) {
      if (specfem::utilities::is_hdf5_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_reader<
            DimensionTag, specfem::io_backends::HDF5>>(
            this->output_folder, this->time_interval, this->include_last_step);
      } else if (specfem::utilities::is_adios2_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_reader<
            DimensionTag, specfem::io_backends::ADIOS2>>(
            this->output_folder, this->time_interval, this->include_last_step);
      } else if (specfem::utilities::is_ascii_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_reader<
            DimensionTag, specfem::io_backends::ASCII>>(
            this->output_folder, this->time_interval, this->include_last_step);
      } else if (specfem::utilities::is_npy_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_reader<
            DimensionTag, specfem::io_backends::NPY>>(
            this->output_folder, this->time_interval, this->include_last_step);
      } else if (specfem::utilities::is_npz_string(this->output_format)) {
        return std::make_shared<specfem::periodic_tasks::wavefield_reader<
            DimensionTag, specfem::io_backends::NPZ>>(
            this->output_folder, this->time_interval, this->include_last_step);
      } else {
        throw std::runtime_error("Unknown wavefield format");
      }
    } else {
      return nullptr;
    }
  }();

  return reader;
}

// Explicit template instantiations
template std::shared_ptr<specfem::periodic_tasks::periodic_task<
    specfem::element::dimension_tag::dim2>>
specfem::runtime_configuration::wavefield::instantiate_wavefield_writer<
    specfem::element::dimension_tag::dim2>() const;

template std::shared_ptr<specfem::periodic_tasks::periodic_task<
    specfem::element::dimension_tag::dim2>>
specfem::runtime_configuration::wavefield::instantiate_wavefield_reader<
    specfem::element::dimension_tag::dim2>() const;

template std::shared_ptr<specfem::periodic_tasks::periodic_task<
    specfem::element::dimension_tag::dim3>>
specfem::runtime_configuration::wavefield::instantiate_wavefield_writer<
    specfem::element::dimension_tag::dim3>() const;

template std::shared_ptr<specfem::periodic_tasks::periodic_task<
    specfem::element::dimension_tag::dim3>>
specfem::runtime_configuration::wavefield::instantiate_wavefield_reader<
    specfem::element::dimension_tag::dim3>() const;
