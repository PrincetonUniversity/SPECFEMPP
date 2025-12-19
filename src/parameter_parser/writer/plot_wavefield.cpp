#include "parameter_parser/writer/plot_wavefield.hpp"
#include "enumerations/display.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/periodic_tasks.hpp"
#include "specfem_mpi/interface.hpp"
#include "utilities/strings.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::plot_wavefield::plot_wavefield(
    const YAML::Node &Node) {

  const std::string output_format = [&]() -> std::string {
    if (Node["format"]) {
      return Node["format"].as<std::string>();
    } else {
      return "PNG";
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

  const std::string component = [&]() -> std::string {
    if (Node["field"]) {
      return Node["field"].as<std::string>();
    } else {
      throw std::runtime_error(
          "Plotting wavefield not specified in the display section");
    }
  }();

  const std::string wavefield_type = [&]() -> std::string {
    if (Node["simulation-field"]) {
      return Node["simulation-field"].as<std::string>();
    } else {
      throw std::runtime_error(
          "Simulation field type not specified in the display section");
    }
  }();

  const int time_interval = [&]() -> int {
    if (Node["time-interval"]) {
      return Node["time-interval"].as<int>();
    } else {
      throw std::runtime_error(
          "Time interval not specified in the display section");
    }
  }();

  *this = specfem::runtime_configuration::plot_wavefield(
      output_format, output_folder, component, wavefield_type, time_interval);

  return;
}

template <specfem::dimension::type DimensionTag>
std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag> >
specfem::runtime_configuration::plot_wavefield::instantiate_wavefield_plotter(
    const specfem::assembly::assembly<DimensionTag> &assembly,
    const type_real &dt, specfem::MPI::MPI *mpi) const {

  const auto output_format = [&]() {
    if (specfem::utilities::is_png_string(this->output_format)) {
      return specfem::display::format::PNG;
    } else if (specfem::utilities::is_jpg_string(this->output_format)) {
      return specfem::display::format::JPG;
    } else if (specfem::utilities::is_onscreen_string(this->output_format)) {
      return specfem::display::format::on_screen;
    } else if (specfem::utilities::is_vtkhdf_string(this->output_format)) {
      return specfem::display::format::vtkhdf;
    } else {
      throw std::runtime_error("Unknown plotter format");
    }
  }();

  const auto component = [&]() {
    if (specfem::utilities::is_displacement_string(this->component)) {
      return specfem::wavefield::type::displacement;
    } else if (specfem::utilities::is_velocity_string(this->component)) {
      return specfem::wavefield::type::velocity;
    } else if (specfem::utilities::is_acceleration_string(this->component)) {
      return specfem::wavefield::type::acceleration;
    } else if (specfem::utilities::is_pressure_string(this->component)) {
      return specfem::wavefield::type::pressure;
    } else if (specfem::utilities::is_rotation_string(this->component)) {
      return specfem::wavefield::type::rotation;
    } else if (specfem::utilities::is_intrinsic_rotation_string(
                   this->component)) {
      return specfem::wavefield::type::intrinsic_rotation;
    } else if (specfem::utilities::is_curl_string(this->component)) {
      return specfem::wavefield::type::curl;
    } else {
      throw std::runtime_error(
          "Unknown wavefield component in the display section");
    }
  }();

  const auto wavefield = [&]() {
    if (specfem::utilities::is_forward_string(this->wavefield_type)) {
      return specfem::wavefield::simulation_field::forward;
    } else if (specfem::utilities::is_adjoint_string(this->wavefield_type)) {
      return specfem::wavefield::simulation_field::adjoint;
    } else if (specfem::utilities::is_backward_string(this->wavefield_type)) {
      return specfem::wavefield::simulation_field::backward;
    } else {
      throw std::runtime_error("Unknown wavefield type in the display section");
    }
  }();

  return std::make_shared<
      specfem::periodic_tasks::plot_wavefield<DimensionTag> >(
      assembly, output_format, component, wavefield, dt, time_interval,
      this->output_folder, mpi);
}

// Explicit template instantiations
template std::shared_ptr<
    specfem::periodic_tasks::periodic_task<specfem::dimension::type::dim2> >
specfem::runtime_configuration::plot_wavefield::instantiate_wavefield_plotter<
    specfem::dimension::type::dim2>(
    const specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const type_real &dt, specfem::MPI::MPI *mpi) const;

template std::shared_ptr<
    specfem::periodic_tasks::periodic_task<specfem::dimension::type::dim3> >
specfem::runtime_configuration::plot_wavefield::instantiate_wavefield_plotter<
    specfem::dimension::type::dim3>(
    const specfem::assembly::assembly<specfem::dimension::type::dim3> &assembly,
    const type_real &dt, specfem::MPI::MPI *mpi) const;
