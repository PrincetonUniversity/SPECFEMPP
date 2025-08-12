#include "mortar_fixtures.hpp"
#include "enumerations/dimension.hpp"
#include "io/interface.hpp"
#include <fstream>
#include <memory>
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

std::vector<std::string>
test_configuration::meshes_in_test(const std::string &testname) {
  if (YAML_DATA.IsNull()) {
    YAML_DATA = YAML::LoadFile(config_filename);
  }
  std::vector<std::string> meshes;
  YAML::Node meshlist = YAML_DATA[testname];
  assert(meshlist.IsSequence());
  for (const auto &meshentry : meshlist) {
    meshes.push_back(meshentry["mesh"].as<std::string>());
  }
  return meshes;
}

MESHES::MESHES() {
  for (const auto &mesh : get_meshes())
    this->push_back(mesh);
}

const specfem::mesh::mesh<specfem::dimension::type::dim2> &
test_configuration::mesh::get_mesh() const {
  return mesh_instance;
}

test_configuration::mesh::mesh(const YAML::Node &node) {
  name = node["name"].as<std::string>();
  description = node["description"].as<std::string>();
  database = node["database"].as<std::string>();
  if (node["interfaces"]) {
    interface_file = node["interfaces"].as<std::string>();
  } else {
    interface_file = "_NONE_";
  }

  nspec = node["nspec"].as<int>();
  characteristic_length = node["characteristic-length"].as<double>();
  nmat = node["num-materials"].as<int>();

  mesh_instance = specfem::io::read_2d_mesh(
      database, specfem::enums::elastic_wave::psv,
      specfem::enums::electromagnetic_wave::te, MPIEnvironment::get_mpi());

  interface_fluid_2d = false;
  interface_solid_2d = false;
  interface_fluid_fluid_2d = false;
  interface_fluid_solid_2d = false;
  interface_solid_fluid_2d = false;
  interface_solid_solid_2d = false;
  for (const auto &interfacenode : node["interface-containers"]) {
    std::string interfacekind = interfacenode.as<std::string>();
    if (interfacekind == "solid_2d") {
      interface_solid_2d = true;
    } else if (interfacekind == "fluid_2d") {
      interface_fluid_2d = true;
    } else if (interfacekind == "solid_fluid_2d") {
      interface_solid_fluid_2d = true;
    } else if (interfacekind == "fluid_fluid_2d") {
      interface_fluid_fluid_2d = true;
    } else if (interfacekind == "solid_solid_2d") {
      interface_solid_solid_2d = true;
    } else if (interfacekind == "fluid_solid_2d") {
      interface_fluid_solid_2d = true;
    } else {
      throw std::runtime_error(
          std::string("Error with test_config.yaml -- mesh '") + name +
          "' has an unknown interface-containers entry: " + interfacekind);
    }
  }
}
