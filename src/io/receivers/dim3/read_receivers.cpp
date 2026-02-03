// Internal Includes
#include "io/interface.hpp"
#include "specfem/receivers.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"

// External Includes
#include <boost/tokenizer.hpp>
#include <fstream>
#include <string>
#include <vector>

std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::dimension::type::dim3> > >
specfem::io::read_3d_receivers(const std::string &stations_file) {

  boost::char_separator<char> sep(" ");
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::dimension::type::dim3> > >
      receivers;
  std::fstream stations;
  stations.open(stations_file, std::ios::in);
  if (stations.is_open()) {
    std::string line;
    // Read stations file line by line
    while (std::getline(stations, line)) {
      // split every line with " " delimiter
      boost::tokenizer<boost::char_separator<char> > tokens(line, sep);
      std::vector<std::string> current_station;
      for (const auto &t : tokens) {
        current_station.push_back(t);
      }
      // check if the read line meets the format
      assert(current_station.size() == 6);
      /* Get the network and station name
       * 3D format: station, network, y, x, elevation, z
       */
      const std::string station_name = current_station[0];
      const std::string network_name = current_station[1];
      // get the x, y and z coordinates of the station; Note the switch in the
      // columns of x and y. This is due to the latitude/longitude convention,
      // where the y coordinate is the latitude and the x coordinate is the
      // longitude.
      const type_real y = static_cast<type_real>(std::stod(current_station[2]));
      const type_real x = static_cast<type_real>(std::stod(current_station[3]));
      // elevation is current_station[4] - not used for receiver position
      const type_real z = static_cast<type_real>(std::stod(current_station[5]));

      receivers.push_back(
          std::make_shared<
              specfem::receivers::receiver<specfem::dimension::type::dim3> >(
              network_name, station_name, x, y, z));
    }

    stations.close();
  }

  // Warn if no receivers were found
  if (receivers.empty()) {
    std::cout << "\033[1mWARNING: No receiver stations found in the STATIONS "
                 "file\033[0m"
              << std::endl;
  }

  return receivers;
}

std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::dimension::type::dim3> > >
specfem::io::read_3d_receivers(const YAML::Node &stations) {

  // If stations file is a string then read the stations file from text format
  try {
    std::string stations_file = stations["stations"].as<std::string>();
    return read_3d_receivers(stations_file);
  } catch (const YAML::Exception &e) {
    // If stations file is not a string then read the stations from the YAML
    // node
  }

  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::dimension::type::dim3> > >
      receivers;

  // Throw error if length of stations is zero or if it is not a sequence
  if (stations["stations"].IsSequence()) {
    if (stations["stations"].size() == 0) {
      throw std::runtime_error("No receiver stations found in the YAML file");
    }
  } else {
    throw std::runtime_error(
        "Expected stations to be a YAML node sequence,\n but it is "
        "neither a sequence nor text file");
  }

  try {
    for (const auto &station : stations["stations"]) {
      const std::string network_name = station["network"].as<std::string>();
      const std::string station_name = station["station"].as<std::string>();
      const type_real x = station["x"].as<type_real>();
      const type_real y = station["y"].as<type_real>();
      const type_real z = station["z"].as<type_real>();

      receivers.push_back(
          std::make_shared<
              specfem::receivers::receiver<specfem::dimension::type::dim3> >(
              network_name, station_name, x, y, z));
    }
  } catch (const YAML::Exception &e) {
    std::cerr << e.what() << std::endl;
    throw std::runtime_error("Error reading receiver stations");
  }

  return receivers;
}
