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
