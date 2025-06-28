#include "mortar_fixtures.hpp"
#include <fstream>
#include <regex>
#include <stdexcept>
#include <utility>
#include <vector>

YAML::Node YAML_DATA = YAML::Node();
static const std::string config_filename = "mortar/test_config.yaml";

static std::vector<test_configuration::mesh> get_meshes() {
  if (YAML_DATA.IsNull()) {
    YAML_DATA = YAML::LoadFile(config_filename);
  }
  std::vector<test_configuration::mesh> meshes;
  YAML::Node meshlist = YAML_DATA["meshes"];
  assert(meshlist.IsSequence());
  for (const auto &meshentry : meshlist) {
    meshes.push_back(test_configuration::mesh(meshentry));
  }
  return meshes;
}

MESHES::MESHES() {
  for (const auto &mesh : get_meshes())
    this->push_back(mesh);
}

static specfem::enums::edge::type edge_from_code(const std::string &st) {
  if (st == "T") {
    return specfem::enums::edge::type::TOP;
  } else if (st == "B") {
    return specfem::enums::edge::type::BOTTOM;
  } else if (st == "L") {
    return specfem::enums::edge::type::LEFT;
  } else if (st == "R") {
    return specfem::enums::edge::type::RIGHT;
  }
  return specfem::enums::edge::type::NONE;
}

specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
test_configuration::mesh::reference_adjacency_map() const {
  specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
      adjmap(nspec);

  std::ifstream adjstream(correct_adjacencies);
  if (!adjstream.is_open()) {
    throw std::runtime_error(
        "Failed to open adjacency provenance file for mesh \"" + name + "\".");
  }

  std::string line;
  std::regex conforming_edge_regex("C\\s*(\\d+)([TBLR])\\s*(\\d+)([TBLR])");
  std::smatch match;
  while (std::getline(adjstream, line)) {
    // parse ispec and edge
    try {
      if (std::regex_search(line, match, conforming_edge_regex)) {
        adjmap.create_conforming_adjacency(
            std::stoi(match[1].str()), edge_from_code(match[2].str()),
            std::stoi(match[3].str()), edge_from_code(match[4].str()));
      }
    } catch (std::runtime_error &e) {
      std::cout << "Error parsing adjacency provenance line \"" << line << "\""
                << std::endl;
      throw;
    }
  }

  adjstream.close();
  return adjmap;
}

std::vector<std::set<std::pair<int, specfem::enums::boundaries::type> > >
test_configuration::mesh::reference_corner_groups() const {
  std::ifstream assem_stream(correct_corner_groups);
  if (!assem_stream.is_open()) {
    throw std::runtime_error(
        "Failed to open assembly provenance file for mesh \"" + name + "\".");
  }
  std::vector<std::set<std::pair<int, specfem::enums::boundaries::type> > >
      groups;

  std::string line;
  std::regex index_regex("(\\d+)([TB])([LR])");

  while (std::getline(assem_stream, line)) {
    std::set<std::pair<int, specfem::enums::boundaries::type> > inds;
    std::sregex_iterator end = std::sregex_iterator();
    for (std::sregex_iterator it(line.begin(), line.end(), index_regex);
         it != end; it++) {
      auto match = *it;
      specfem::enums::boundaries::type corner;
      if (match[2] == 'T') {
        if (match[3] == 'L') {
          corner = specfem::enums::boundaries::type::TOP_LEFT;
        } else {
          corner = specfem::enums::boundaries::type::TOP_RIGHT;
        }
      } else {
        if (match[3] == 'L') {
          corner = specfem::enums::boundaries::type::BOTTOM_LEFT;
        } else {
          corner = specfem::enums::boundaries::type::BOTTOM_RIGHT;
        }
      }

      inds.insert(std::make_pair(std::stoi(match[1]), corner));
    }
    groups.push_back(inds);
  }

  assem_stream.close();
  return groups;
}
