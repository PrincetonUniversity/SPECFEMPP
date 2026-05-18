#include "specfem/io/seismogram/impl/channel_generator.hpp"
#include "specfem/enums.hpp"
#include <iostream>
#include <span>
#include <string>
#include <vector>

std::string
specfem::io::impl::ChannelGenerator::compute_band_code(const type_real dt) {

  // Round to nearest Hz to avoid float boundary issues
  // (e.g. dt=0.1f → 9.9999998 Hz without rounding, which would fall in M).
  // FDSN band boundaries are exact integers (1, 10, 80, 250, 1000 Hz).
  const double fs = std::round(1.0 / static_cast<double>(dt));

  // see header for lengthy discourse on band codes and reference
  if (fs < 1.0) {
    return "L"; // ~1 Hz (long period)
  } else if (fs < 10.0) {
    return "M"; // > 1 to < 10 Hz
  } else if (fs < 80.0) {
    return "B"; // >= 10 to < 80 Hz
  } else if (fs < 250.0) {
    return "H"; // >= 80 to < 250 Hz
  } else if (fs < 1000.0) {
    return "C"; // >= 250 to < 1000 Hz
  } else {
    return "F"; // >= 1000 Hz
  }
}

std::string specfem::io::impl::ChannelGenerator::get_channel_code(
    const char component_letter) {
  std::string result = this->band_code + "X" + std::string(1, component_letter);
  return result;
}

std::string specfem::io::impl::ChannelGenerator::get_file_extension(
    specfem::enums::wavefield seismogram_type) {
  switch (seismogram_type) {
  case specfem::enums::wavefield::displacement:
    return "semd";
  case specfem::enums::wavefield::velocity:
    return "semv";
  case specfem::enums::wavefield::acceleration:
    return "sema";
  case specfem::enums::wavefield::pressure:
    return "semp";
  case specfem::enums::wavefield::rotation:
    return "semr";
  case specfem::enums::wavefield::intrinsic_rotation:
    return "semir";
  case specfem::enums::wavefield::curl:
    return "semc";
  default:
    throw std::runtime_error("Unknown seismogram type for file extension.");
  }
}

std::vector<std::string>
specfem::io::impl::ChannelGenerator::get_station_filenames(
    const std::string &network_name, const std::string &station_name,
    const std::string &location_code,
    const specfem::enums::wavefield seismogram_type,
    std::span<const char> elastic_components) {

  std::string channel_code;
  std::vector<std::string> filenames;

  // Catch to hanndle empty location code
  std::string location_code_with_dot = "";
  if (!location_code.empty()) {
    location_code_with_dot = location_code + ".";
  }

  // Create a filename vector depending on seismogram type
  switch (seismogram_type) {
  case specfem::enums::wavefield::displacement:
  case specfem::enums::wavefield::velocity:
  case specfem::enums::wavefield::acceleration:
    // TODO (Lucas : This will need to be modified once we allow different
    //               coordinates systems, e.g., NEZ, RTZ, etc.)
    for (const auto &component_letter : elastic_components) {

      // Get the channel code based on component and timestep
      channel_code = this->get_channel_code(component_letter);

      // TODO (Lucas : CPP20 std::format would be perfect here)
      // Get the filename for the current component
      filenames.push_back(network_name + "." + station_name + "." +
                          location_code_with_dot + channel_code + "." +
                          this->get_file_extension(seismogram_type));
    }
    break;

  case specfem::enums::wavefield::rotation:

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

  case specfem::enums::wavefield::intrinsic_rotation:

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

  case specfem::enums::wavefield::curl:

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

  case specfem::enums::wavefield::pressure:

    channel_code = this->get_channel_code('P');
    filenames = { network_name + "." + station_name + "." +
                  location_code_with_dot + channel_code + "." +
                  this->get_file_extension(seismogram_type) };
    break;
  default:
    std::ostringstream message;
    message << "Error reading specfem receiver configuration. (" << __FILE__
            << ":" << __LINE__ << ")\n";
    message << "Unknown seismogram type: "
            << specfem::enums::to_string(seismogram_type) << "\n";
    message << "Valid seismogram types are: displacement, velocity, "
            << "acceleration, pressure, rotation, intrinsic_rotation, curl.\n";
    message << "Please check your configuration file.\n";
    throw std::runtime_error(message.str());
  }

  return filenames;
}
