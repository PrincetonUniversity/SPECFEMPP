#include "parameter_parser/receivers.hpp"
#include "constants.hpp"
#include "yaml-cpp/yaml.h"

// External Includes
#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

// specfem::runtime_configuration::receivers::receivers(
//     const std::string stations_file, const type_real angle,
//     const int nstep_between_samples) {

//   // Define basic parameters
//   this->angle = angle;
//   this->nstep_between_samples = nstep_between_samples;

//   // Open file stream
//   std::fstream stations;

//   // Define separator
//   boost::char_separator<char> sep(" ");

//   // Create empty sequence Node
//   this->stations_node = YAML::Load("[]");

//   // Open stations file
//   stations.open(stations_file, std::ios::in);

//   if (stations.is_open()) {

//     std::string line;

//     // Read stations file line by line
//     while (std::getline(stations, line)) {

//       // split every line with " " delimiter
//       boost::tokenizer<boost::char_separator<char> > tokens(line, sep);

//       // Create a vector to store the current station
//       std::vector<std::string> current_station;

//       for (const auto &t : tokens) {
//         current_station.push_back(t);
//       }

//       // check if the read line meets the format
//       assert(current_station.size() == 6);

//       // get the x and z coordinates of the station;
//       const std::string network_name = current_station[0];
//       const std::string station_name = current_station[1];
//       const std::string x = current_station[2];
//       const std::string z = current_station[3];

//       // Current station node
//       YAML::Node current_station_node;

//       // Edit node to include the current station
//       current_station_node["network"] = network_name;
//       current_station_node["station"] = station_name;
//       current_station_node["x"] = x;
//       current_station_node["z"] = z;

//       // Append current station to the stations node
//       this->stations_node.push_back(current_station_node);
//     }
//   } else {
//     std::ostringstream message;
//     message << "Could not open stations file: " << stations_file << "\n";
//     message << "Receivers empty.";
//     return;
//   }
// }

// specfem::runtime_configuration::receivers::receivers(const YAML::Node &Node)
// {
//   try {
//     if (const YAML::Node &n_stations = Node["stations-file"]) {
//       *this = specfem::runtime_configuration::receivers(
//           n_stations.as<std::string>(), Node["angle"].as<type_real>(),
//           Node["nstep_between_samples"].as<int>());
//     } else if (const YAML::Node &n_stations = Node["stations-dict"]) {
//       *this = specfem::runtime_configuration::receivers(
//           n_stations, Node["angle"].as<type_real>(),
//           Node["nstep_between_samples"].as<int>());
//     } else {
//       throw std::runtime_error("Error reading specfem receiver
//       configuration.");
//     }
//   } catch (YAML::ParserException &e) {
//     std::ostringstream message;

//     message << "Error reading specfem receiver configuration. \n" <<
//     e.what();

//     throw std::runtime_error(message.str());
//   }
// }

std::vector<specfem::enums::seismogram::type>
specfem::runtime_configuration::receivers::get_seismogram_types() const {

  std::vector<specfem::enums::seismogram::type> stypes;

  // Allocate seismogram types
  assert(this->receivers_node["seismogram-type"].IsSequence());

  for (YAML::Node seismogram_type : this->receivers_node["seismogram-type"]) {
    if (seismogram_type.as<std::string>() == "displacement") {
      stypes.push_back(specfem::enums::seismogram::type::displacement);
    } else if (seismogram_type.as<std::string>() == "velocity") {
      stypes.push_back(specfem::enums::seismogram::type::velocity);
    } else if (seismogram_type.as<std::string>() == "acceleration") {
      stypes.push_back(specfem::enums::seismogram::type::acceleration);
    } else if (seismogram_type.as<std::string>() == "pressure") {
      stypes.push_back(specfem::enums::seismogram::type::pressure);
    } else {
      std::ostringstream message;

      message << "Error reading specfem receiver configuration. \n";
      message << "Unknown seismogram type: "
              << seismogram_type.as<std::string>() << "\n";

      std::runtime_error(message.str());
    }
  }
  return stypes;
}
