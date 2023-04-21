#include "../include/read_sources.h"
#include "../include/receiver.h"
#include "../include/source.h"
#include "../include/specfem_setup.hpp"
#include "../include/utils.h"
#include "yaml-cpp/yaml.h"
#include <boost/tokenizer.hpp>
#include <fstream>
#include <string>
#include <vector>

// void operator>>(YAML::Node &Node,
//                 specfem::utilities::force_source &force_source) {
//   force_source.x = Node["x"].as<type_real>();
//   force_source.z = Node["z"].as<type_real>();
//   force_source.source_surf = Node["source_surf"].as<bool>();
//   force_source.stf_type = Node["stf_type"].as<std::string>();
//   force_source.f0 = Node["f0"].as<type_real>();
//   force_source.angle = Node["angle"].as<type_real>();
//   force_source.vx = Node["vx"].as<type_real>();
//   force_source.vz = Node["vz"].as<type_real>();
//   force_source.factor = Node["factor"].as<type_real>();
//   force_source.tshift = Node["tshift"].as<type_real>();
// }

// void operator>>(YAML::Node &Node,
//                 specfem::utilities::moment_tensor &moment_tensor) {
//   moment_tensor.x = Node["x"].as<type_real>();
//   moment_tensor.z = Node["z"].as<type_real>();
//   moment_tensor.source_surf = Node["source_surf"].as<bool>();
//   moment_tensor.stf_type = Node["stf_type"].as<std::string>();
//   moment_tensor.f0 = Node["f0"].as<type_real>();
//   moment_tensor.Mxx = Node["Mxx"].as<type_real>();
//   moment_tensor.Mxz = Node["Mxz"].as<type_real>();
//   moment_tensor.Mzz = Node["Mzz"].as<type_real>();
//   moment_tensor.vx = Node["vx"].as<type_real>();
//   moment_tensor.vz = Node["vz"].as<type_real>();
//   moment_tensor.factor = Node["factor"].as<type_real>();
//   moment_tensor.tshift = Node["tshift"].as<type_real>();
// }

std::tuple<std::vector<specfem::sources::source *>, type_real>
specfem::read_sources(const std::string sources_file, const type_real dt,
                      const specfem::MPI::MPI *mpi) {
  // read sources file
  std::vector<specfem::sources::source *> sources;
  YAML::Node yaml = YAML::LoadFile(sources_file);
  int nsources = yaml["number-of-sources"].as<int>();
  YAML::Node Node = yaml["sources"];
  assert(Node.IsSequence());
  for (auto N : Node) {
    if (YAML::Node force_source = N["force"]) {
      sources.push_back(new specfem::sources::force(force_source, dt));
    } else if (YAML::Node moment_tensor = N["moment-tensor"]) {
      sources.push_back(new specfem::sources::moment_tensor(moment_tensor, dt));
    }
  }

  if (sources.size() != nsources) {
    std::ostringstream message;
    message << "Found only " << sources.size()
            << " number of sources. Total number of sources in " << sources_file
            << " are" << nsources
            << " Please check if there is a error in sources file.";
    throw std::runtime_error(message.str());
  }

  type_real t0 = std::numeric_limits<type_real>::max();
  for (auto &source : sources) {
    type_real cur_t0 = source->get_t0();
    if (cur_t0 < t0) {
      t0 = cur_t0;
    }
  }

  for (auto &source : sources) {
    type_real cur_t0 = source->get_t0();
    source->update_tshift(cur_t0 - t0);
  }

  return std::make_tuple(sources, t0);
}

std::vector<specfem::receivers::receiver *>
specfem::read_receivers(const std::string stations_file,
                        const type_real angle) {

  boost::char_separator<char> sep(" ");
  std::vector<specfem::receivers::receiver *> receivers;
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

      receivers.push_back(new specfem::receivers::receiver(
          network_name, station_name, x, z, angle));
    }

    stations.close();
  }

  return receivers;
}
