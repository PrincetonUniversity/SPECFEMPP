#pragma once

#include "../../MPI_environment.hpp"
#include "enumerations/interface.hpp"
#include "io/interface.hpp"
#include "mesh/mesh.hpp"
#include "quadrature/quadratures.hpp"
#include "source/source.hpp"
#include "specfem/assembly.hpp"
#include "specfem/receivers.hpp"
#include "utilities/strings.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

template <bool using_simd>
KOKKOS_FUNCTION
    specfem::point::index<specfem::dimension::type::dim2, using_simd>
    get_index(const int ielement, const int num_elements, const int iz,
              const int ix);

// ------------------------------------------------------------------------
// Test configuration
namespace test_configuration {
struct database {
public:
  database() : mesh(""), sources(""), stations("") {};
  database(const YAML::Node &Node) {
    mesh = Node["mesh"].as<std::string>();
    sources = Node["sources"].as<std::string>();
    stations = Node["stations"].as<std::string>();
  }

  std::tuple<std::string, std::string, std::string> get_databases() {
    return std::make_tuple(mesh, sources, stations);
  }

  std::string mesh;
  std::string sources;
  std::string stations;
};

struct config {
public:
  config() : nproc(1), elastic_wave("P_SV"), electromagnetic_wave("TE") {};
  config(const YAML::Node &Node) {
    nproc = Node["nproc"].as<int>();
    elastic_wave = Node["elastic_wave"].as<std::string>();

    if (Node["electromagnetic_wave"].IsDefined())
      electromagnetic_wave = Node["electromagnetic_wave"].as<std::string>();
    else
      // Default to TE if not defined
      electromagnetic_wave = "TE";
  }

  int get_nproc() { return nproc; }

  specfem::enums::elastic_wave get_elastic_wave() {
    if (specfem::utilities::is_psv_string(elastic_wave))
      return specfem::enums::elastic_wave::psv;
    else if (specfem::utilities::is_sh_string(elastic_wave))
      return specfem::enums::elastic_wave::sh;
    else
      throw std::runtime_error("Elastic wave type not supported");
  }

  specfem::enums::electromagnetic_wave get_electromagnetic_wave() {
    if (specfem::utilities::is_te_string(electromagnetic_wave))
      return specfem::enums::electromagnetic_wave::te;
    else if (specfem::utilities::is_tm_string(electromagnetic_wave))
      return specfem::enums::electromagnetic_wave::tm;
    else
      throw std::runtime_error("Electromagnetic wave type not supported");
  }

private:
  int nproc;
  std::string elastic_wave;
  std::string electromagnetic_wave;
};

struct Test {
public:
  Test(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    suffix = Node["suffix"].as<std::string>();
    YAML::Node databases = Node["databases"];
    YAML::Node configuration = Node["config"];
    try {
      database = test_configuration::database(databases);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Error in test configuration: " + name + "\n" +
                               e.what());
    }
    try {
      config = test_configuration::config(configuration);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Error in test configuration: " + name + "\n" +
                               e.what());
    }
    return;
  }

  std::tuple<std::string, std::string, std::string> get_databases() {
    return database.get_databases();
  }

  int get_nproc() { return config.get_nproc(); }

  specfem::enums::elastic_wave get_elastic_wave() {
    return config.get_elastic_wave();
  }

  specfem::enums::electromagnetic_wave get_electromagnetic_wave() {
    return config.get_electromagnetic_wave();
  }

  std::string get_suffix() { return suffix; }

  std::string name;
  std::string description;
  std::string suffix;
  test_configuration::database database;
  test_configuration::config config;
};
} // namespace test_configuration

// ------------------------------------------------------------------------

class ASSEMBLY : public ::testing::Test {

protected:
  class Iterator {
  public:
    Iterator(
        test_configuration::Test *p_Test,
        specfem::mesh::mesh<specfem::dimension::type::dim2> *p_mesh,
        std::vector<std::shared_ptr<specfem::sources::source> > *p_sources,
        std::vector<std::shared_ptr<
            specfem::receivers::receiver<specfem::dimension::type::dim2> > >
            *p_stations,
        std::string *p_suffixes,
        specfem::assembly::assembly<specfem::dimension::type::dim2> *p_assembly)
        : p_Test(p_Test), p_mesh(p_mesh), p_sources(p_sources),
          p_stations(p_stations), p_suffixes(p_suffixes),
          p_assembly(p_assembly) {}

    std::tuple<test_configuration::Test,
               specfem::mesh::mesh<specfem::dimension::type::dim2>,
               std::vector<std::shared_ptr<specfem::sources::source> >,
               std::vector<std::shared_ptr<specfem::receivers::receiver<
                   specfem::dimension::type::dim2> > >,
               std::string,
               specfem::assembly::assembly<specfem::dimension::type::dim2> >
    operator*() {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[RUNNING]\033[0m " << p_Test->name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      return std::make_tuple(*p_Test, *p_mesh, *p_sources, *p_stations,
                             *p_suffixes, *p_assembly);
    }

    Iterator &operator++() {
      ++p_Test;
      ++p_mesh;
      ++p_sources;
      ++p_stations;
      ++p_suffixes;
      ++p_assembly;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return p_Test != other.p_Test;
    }

  private:
    test_configuration::Test *p_Test;
    specfem::mesh::mesh<specfem::dimension::type::dim2> *p_mesh;
    std::vector<std::shared_ptr<specfem::sources::source> > *p_sources;
    std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::dimension::type::dim2> > >
        *p_stations;
    std::string *p_suffixes;
    specfem::assembly::assembly<specfem::dimension::type::dim2> *p_assembly;
  };

  ASSEMBLY();

  Iterator begin() {
    return Iterator(&Tests[0], &Meshes[0], &Sources[0], &Stations[0],
                    &suffixes[0], &assemblies[0]);
  }

  Iterator end() {
    return Iterator(&Tests[Tests.size()], &Meshes[Meshes.size()],
                    &Sources[Sources.size()], &Stations[Stations.size()],
                    &suffixes[suffixes.size()], &assemblies[assemblies.size()]);
  }

  std::vector<test_configuration::Test> Tests;
  std::vector<specfem::mesh::mesh<specfem::dimension::type::dim2> > Meshes;
  std::vector<std::vector<std::shared_ptr<specfem::sources::source> > > Sources;
  std::vector<std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::dimension::type::dim2> > > >
      Stations;
  std::vector<std::string> suffixes;
  std::vector<specfem::assembly::assembly<specfem::dimension::type::dim2> >
      assemblies;
};
