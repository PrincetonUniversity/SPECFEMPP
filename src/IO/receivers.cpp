#include "IO/interface.hpp"
#include "receiver/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
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
