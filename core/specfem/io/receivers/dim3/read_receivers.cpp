// Internal Includes
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/io.hpp"
#include "specfem/receivers.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"

// External Includes
#include <array>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
specfem::io::read_3d_receivers(const std::string &stations_file,
                               bool geographic) {

  boost::char_separator<char> sep(" \t", "", boost::drop_empty_tokens);
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
      receivers;
  std::fstream stations;
  stations.open(stations_file, std::ios::in);
  if (stations.is_open()) {
    std::string line;
    // Read stations file line by line
    while (std::getline(stations, line)) {
      // split every line on any whitespace (spaces or tabs)
      boost::tokenizer<boost::char_separator<char>> tokens(line, sep);
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
      assert(station_name.size() <= 32 &&
             "Station name must be at most 32 characters");
      assert(network_name.size() <= 8 &&
             "Network name must be at most 8 characters");
      // STATIONS columns (SPECFEM3D convention, read_stations.f90):
      //   name network latitude longitude elevation burial
      // Elevation (col 5) is not used for placement; burial (col 6) is depth.
      if (geographic) {
        const double latitude = std::stod(current_station[2]);
        const double longitude = std::stod(current_station[3]);
        const double depth = std::stod(current_station[5]);
        receivers.push_back(std::make_shared<specfem::receivers::receiver<
                                specfem::element::dimension_tag::dim3>>(
            network_name, station_name,
            std::make_unique<
                specfem::coordinate_systems::geographic_coordinates>(
                longitude, latitude, depth)));
      } else {
        // Cartesian: latitude column -> y, longitude column -> x, burial
        // (col 6) -> absolute z. Resolution is deferred to assembly time via
        // read_coordinates_ (origin {0,0,0}: already global).
        const double y = std::stod(current_station[2]);
        const double x = std::stod(current_station[3]);
        const double z = std::stod(current_station[5]);
        receivers.push_back(std::make_shared<specfem::receivers::receiver<
                                specfem::element::dimension_tag::dim3>>(
            network_name, station_name,
            std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
                specfem::element::dimension_tag::dim3>>(
                x, y, z, std::array<double, 3>{ 0.0, 0.0, 0.0 })));
      }
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
    specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
specfem::io::read_3d_receivers(const YAML::Node &stations, bool geographic) {

  // If stations file is a string then read the stations file from text format
  try {
    std::string stations_file = stations["stations"].as<std::string>();
    return read_3d_receivers(stations_file, geographic);
  } catch (const YAML::Exception &e) {
    // If stations file is not a string then read the stations from the YAML
    // node
  }

  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
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

      if (geographic) {
        const double longitude = station["longitude"].as<double>();
        const double latitude = station["latitude"].as<double>();
        const double depth = station["depth"].as<double>(); // meters
        receivers.push_back(std::make_shared<specfem::receivers::receiver<
                                specfem::element::dimension_tag::dim3>>(
            network_name, station_name,
            std::make_unique<
                specfem::coordinate_systems::geographic_coordinates>(
                longitude, latitude, depth)));
      } else if (station["z"]) {
        // Absolute cartesian: origin {0,0,0}.
        receivers.push_back(std::make_shared<specfem::receivers::receiver<
                                specfem::element::dimension_tag::dim3>>(
            network_name, station_name,
            std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
                specfem::element::dimension_tag::dim3>>(
                station["x"].as<double>(), station["y"].as<double>(),
                station["z"].as<double>(),
                std::array<double, 3>{ 0.0, 0.0, 0.0 })));
      } else {
        // Depth-based cartesian (x/y/depth, no z): z = -depth, origin nullopt
        // so it is resolved against topography at assembly time.
        receivers.push_back(std::make_shared<specfem::receivers::receiver<
                                specfem::element::dimension_tag::dim3>>(
            network_name, station_name,
            std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
                specfem::element::dimension_tag::dim3>>(
                station["x"].as<double>(), station["y"].as<double>(),
                -station["depth"].as<double>(), std::nullopt)));
      }
    }
  } catch (const YAML::Exception &e) {
    std::cerr << e.what() << std::endl;
    throw std::runtime_error("Error reading receiver stations");
  }

  return receivers;
}
