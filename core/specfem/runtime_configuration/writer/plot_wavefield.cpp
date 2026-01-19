#include "plot_wavefield.hpp"
#include "enumerations/display.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/periodic_tasks.hpp"

#include "specfem/utilities.hpp"
#include <boost/filesystem.hpp>

specfem::runtime_configuration::plot_wavefield::plot_wavefield(
    const YAML::Node &Node, specfem::enums::elastic_wave elastic_wave,
    specfem::enums::electromagnetic_wave electromagnetic_wave)
    : elastic_wave(elastic_wave), electromagnetic_wave(electromagnetic_wave) {

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

  const std::string field_type = [&]() -> std::string {
    if (Node["field"]) {
      return Node["field"].as<std::string>();
    } else {
      throw std::runtime_error(
          "Plotting wavefield not specified in the display section");
    }
  }();

  const std::string simulation_wavefield_type = [&]() -> std::string {
    if (Node["simulation-field"]) {
      return Node["simulation-field"].as<std::string>();
    } else {
      throw std::runtime_error(
          "Simulation field type not specified in the display section");
    }
  }();

  const std::string component = [&]() -> std::string {
    if (Node["component"]) {
      return Node["component"].as<std::string>();
    } else {
      return "magnitude";
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
      output_format, output_folder, field_type, simulation_wavefield_type,
      component, time_interval, elastic_wave, electromagnetic_wave);

  return;
}

template <specfem::dimension::type DimensionTag>
std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag> >
specfem::runtime_configuration::plot_wavefield::instantiate_wavefield_plotter(
    const specfem::assembly::assembly<DimensionTag> &assembly,
    const type_real &dt) const {

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
    if (specfem::utilities::is_x_string(this->component)) {
      return specfem::display::component::x;
    } else if (specfem::utilities::is_y_string(this->component)) {
      return specfem::display::component::y;
    } else if (specfem::utilities::is_z_string(this->component)) {
      return specfem::display::component::z;
    } else if (specfem::utilities::is_magnitude_string(this->component)) {
      return specfem::display::component::magnitude;
    } else {
      throw std::runtime_error("Unknown plotter component");
    }
  }();

  // Throw error if component is y and elastic wave type is SH
  if constexpr (DimensionTag == specfem::dimension::type::dim2) {
    if (component == specfem::display::component::y &&
        (elastic_wave == specfem::enums::elastic_wave::psv)) {
      std::ostringstream message;
      message
          << "Error: Y component plotting is not supported for P-SV elastic "
             "wave simulations.";
      throw std::runtime_error(message.str());
    }

    // Throw error if component is z or x and elastic wave type is SH
    if ((component == specfem::display::component::x ||
         component == specfem::display::component::z) &&
        (elastic_wave == specfem::enums::elastic_wave::sh)) {
      std::ostringstream message;
      message << "Error: X and Z component plotting is not supported for SH "
                 "elastic wave simulations.";
      throw std::runtime_error(message.str());
    }
  }

  const auto field_type = [&]() {
    if (specfem::utilities::is_displacement_string(this->field_type)) {
      return specfem::wavefield::type::displacement;
    } else if (specfem::utilities::is_velocity_string(this->field_type)) {
      return specfem::wavefield::type::velocity;
    } else if (specfem::utilities::is_acceleration_string(this->field_type)) {
      return specfem::wavefield::type::acceleration;
    } else if (specfem::utilities::is_pressure_string(this->field_type)) {
      return specfem::wavefield::type::pressure;
    } else if (specfem::utilities::is_rotation_string(this->field_type)) {
      return specfem::wavefield::type::rotation;
    } else if (specfem::utilities::is_intrinsic_rotation_string(
                   this->field_type)) {
      return specfem::wavefield::type::intrinsic_rotation;
    } else if (specfem::utilities::is_curl_string(this->field_type)) {
      return specfem::wavefield::type::curl;
    } else {
      throw std::runtime_error(
          "Unknown wavefield component in the display section");
    }
  }();

  const auto simulation_wavefield_type = [&]() {
    if (specfem::utilities::is_forward_string(
            this->simulation_wavefield_type)) {
      return specfem::wavefield::simulation_field::forward;
    } else if (specfem::utilities::is_adjoint_string(
                   this->simulation_wavefield_type)) {
      return specfem::wavefield::simulation_field::adjoint;
    } else if (specfem::utilities::is_backward_string(
                   this->simulation_wavefield_type)) {
      return specfem::wavefield::simulation_field::backward;
    } else {
      throw std::runtime_error("Unknown wavefield type in the display section");
    }
  }();

  if constexpr (DimensionTag == specfem::dimension::type::dim2) {
    return std::make_shared<
        specfem::periodic_tasks::plot_wavefield<DimensionTag> >(
        assembly, output_format, field_type, simulation_wavefield_type,
        component, dt, time_interval, this->output_folder, this->elastic_wave,
        this->electromagnetic_wave);
  } else if constexpr (DimensionTag == specfem::dimension::type::dim3) {
    return std::make_shared<
        specfem::periodic_tasks::plot_wavefield<DimensionTag> >(
        assembly, output_format, field_type, simulation_wavefield_type,
        component, dt, time_interval, this->output_folder);
  }

  throw std::runtime_error("Unsupported dimension for wavefield plotter");
}

// Explicit template instantiations
template std::shared_ptr<
    specfem::periodic_tasks::periodic_task<specfem::dimension::type::dim2> >
specfem::runtime_configuration::plot_wavefield::instantiate_wavefield_plotter<
    specfem::dimension::type::dim2>(
    const specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const type_real &dt) const;

template std::shared_ptr<
    specfem::periodic_tasks::periodic_task<specfem::dimension::type::dim3> >
specfem::runtime_configuration::plot_wavefield::instantiate_wavefield_plotter<
    specfem::dimension::type::dim3>(
    const specfem::assembly::assembly<specfem::dimension::type::dim3> &assembly,
    const type_real &dt) const;
