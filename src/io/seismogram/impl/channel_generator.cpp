#include "io/seismogram/impl/channel_generator.hpp"
#include "enumerations/interface.hpp"
#include <iostream>
#include <string>
#include <vector>

std::string
specfem::io::impl::ChannelGenerator::compute_band_code(const type_real dt) {

  std::string instrument_code = "";

  // see header for lengthy discourse on band codes and reference
  if (dt >= 1.0) {
    instrument_code = "L";
  } else if (dt >= 0.1) {
    instrument_code = "M";
  } else if (dt > 0.0125) {
    instrument_code = "B";
  } else if (dt > 0.004) {
    instrument_code = "H";
  } else if (dt > 0.001) {
    instrument_code = "C";
  } else {
    instrument_code = "F";
  }
  return instrument_code;
}

std::string specfem::io::impl::ChannelGenerator::get_channel_code(
    const char component_letter) {
  std::string result = this->band_code + "X" + std::string(1, component_letter);
  return result;
}

std::string specfem::io::impl::ChannelGenerator::get_file_extension(
    specfem::wavefield::type seismogram_type) {
  switch (seismogram_type) {
  case specfem::wavefield::type::displacement:
    return "semd";
  case specfem::wavefield::type::velocity:
    return "semv";
  case specfem::wavefield::type::acceleration:
    return "sema";
  case specfem::wavefield::type::pressure:
    return "semp";
  default:
    throw std::runtime_error("Unknown seismogram type for file extension.");
  }
}

std::vector<std::string>
specfem::io::impl::ChannelGenerator::get_station_filenames(
    const std::string &network_name, const std::string &station_name,
    const std::string &location_code,
    const specfem::wavefield::type seismogram_type) {

  std::string channel_code;
  std::vector<std::string> filenames;

  // Catch to hanndle empty location code
  std::string location_code_with_dot = "";
  if (!location_code.empty()) {
    location_code_with_dot = location_code + ".";
  }

  // Component letters for 3D seismograms
  // TODO (Lucas : This will need to be modified once we allow different
  //               coordinates systems, e.g., NEZ, RTZ, etc.)
  std::array<char, 3> component_letters = { 'X', 'Y', 'Z' };

  // Create a filename vector depending on seismogram type
  switch (seismogram_type) {
  case specfem::wavefield::type::displacement:
    for (const auto &component_letter : component_letters) {

      // Get the channel code based on component and timestep
      channel_code = this->get_channel_code(component_letter);

      // TODO (Lucas : CPP20 std::format would be perfect here)
      // Get the filename for the current component
      filenames.push_back(output_folder + "/" + network_name + "." +
                          station_name + "." + location_code_with_dot +
                          channel_code + "." +
                          this->get_file_extension(seismogram_type));
    }
    break;

  case specfem::wavefield::type::velocity:

    for (const auto &component_letter : component_letters) {

      // Get the channel code based on component and timestep
      channel_code = this->get_channel_code(component_letter);

      // TODO (Lucas : CPP20 std::format would be perfect here)
      // Get the filename for the current component
      filenames.push_back(output_folder + "/" + network_name + "." +
                          station_name + "." + location_code_with_dot +
                          channel_code + "." +
                          this->get_file_extension(seismogram_type));
    }
    break;

  case specfem::wavefield::type::acceleration:

    for (const auto &component_letter : component_letters) {

      // Get the channel code based on component and timestep
      channel_code = this->get_channel_code(component_letter);

      // Get the filename for the current component
      filenames.push_back(output_folder + "/" + network_name + "." +
                          station_name + "." + location_code_with_dot +
                          channel_code + "." +
                          this->get_file_extension(seismogram_type));
    }
    break;

  case specfem::wavefield::type::pressure:

    channel_code = this->get_channel_code('P');
    filenames = { output_folder + "/" + network_name + "." + station_name +
                  "." + location_code_with_dot + channel_code + "." +
                  this->get_file_extension(seismogram_type) };
    break;
  default:
    std::ostringstream message;
    message << "Error reading specfem receiver configuration. (" << __FILE__
            << ":" << __LINE__ << ")\n";
    message << "Unknown seismogram type: "
            << specfem::wavefield::to_string(seismogram_type) << "\n";
    message << "Valid seismogram types are: displacement, velocity, "
            << "acceleration, pressure, rotation, intrinsic_rotation, curl.\n";
    message << "Please check your configuration file.\n";
    throw std::runtime_error(message.str());
  }

  return filenames;
}
