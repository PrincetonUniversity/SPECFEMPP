// Internal Includes
#include "IO/interface.hpp"
#include "receiver/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"

// External Includes
#include <boost/tokenizer.hpp>
#include <fstream>
#include <string>
#include <vector>

std::vector<std::shared_ptr<specfem::receivers::receiver> >
specfem::IO::read_receivers(const std::string stations_file,
                            const type_real angle) {

  boost::char_separator<char> sep(" ");
  std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers;
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
      // get the x and z coordinates of the station;
      const std::string network_name = current_station[0];
      const std::string station_name = current_station[1];
      const type_real x = static_cast<type_real>(std::stod(current_station[2]));
      const type_real z = static_cast<type_real>(std::stod(current_station[3]));

      receivers.push_back(std::make_shared<specfem::receivers::receiver>(
          network_name, station_name, x, z, angle));
    }

    stations.close();
  }

  return receivers;
}

std::vector<std::shared_ptr<specfem::receivers::receiver> >
specfem::IO::read_receivers(const YAML::Node &stations, const type_real angle) {

  std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers;

  // Throw error if length of stations is zero or if it is not a sequence
  if (stations.IsSequence()) {
    if (stations.size() == 0) {
      throw std::runtime_error("No receiver stations found in the YAML file");
    }
  } else {
    throw std::runtime_error("Receiver YAML node is not a sequence");
  }

  try {
    for (const auto &station : stations) {
      const std::string network_name = station["network"].as<std::string>();
      const std::string station_name = station["station"].as<std::string>();
      const type_real x = station["x"].as<type_real>();
      const type_real z = station["z"].as<type_real>();

      receivers.push_back(std::make_shared<specfem::receivers::receiver>(
          network_name, station_name, x, z, angle));
    }
  } catch (const YAML::Exception &e) {
    std::cerr << e.what() << std::endl;
    throw std::runtime_error("Error reading receiver stations");
  }

  return receivers;
}
