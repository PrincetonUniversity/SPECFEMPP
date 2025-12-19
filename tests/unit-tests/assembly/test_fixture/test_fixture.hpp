#pragma once

#include "../../SPECFEM_Environment.hpp"
#include "enumerations/interface.hpp"
#include "io/interface.hpp"
#include "mesh/mesh.hpp"
#include "quadrature/quadratures.hpp"
#include "specfem/assembly.hpp"
#include "specfem/receivers.hpp"
#include "specfem/source.hpp"
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

template <bool using_simd>
KOKKOS_FUNCTION
    specfem::point::index<specfem::dimension::type::dim3, using_simd>
    get_index(const int ielement, const int num_elements, const int iz,
              const int iy, const int ix);

// ------------------------------------------------------------------------
// Test configuration
namespace test_configuration {

// Base database struct
template <specfem::dimension::type DimensionType> struct database {
public:
  database() : sources(""), stations("") {};
  virtual ~database() = default;

  std::string sources;
  std::string stations;
};

// 2D database specialization
template <> struct database<specfem::dimension::type::dim2> {
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

// 3D database specialization
template <> struct database<specfem::dimension::type::dim3> {
public:
  database()
      : mesh_database(""), mesh_parameters(""), sources(""), stations("") {};
  database(const YAML::Node &Node) {
    mesh_database = Node["mesh-database"].as<std::string>();
    mesh_parameters = Node["mesh-parameters"].as<std::string>();
    sources = Node["sources"].as<std::string>();
    stations = Node["stations"].as<std::string>();
  }

  std::tuple<std::string, std::string, std::string> get_databases() {
    return std::make_tuple(mesh_parameters, mesh_database, sources);
  }

  std::string mesh_database;
  std::string mesh_parameters;
  std::string sources;
  std::string stations;
};

struct source_solution {
public:
  source_solution() {};
  source_solution(type_real xi, type_real gamma, int ispec,
                  specfem::element::medium_tag medium_tag)
      : xi(xi), gamma(gamma), ispec(ispec), medium_tag(medium_tag) {};

  source_solution(const YAML::Node &Node) {
    this->xi = Node["xi"].as<type_real>();
    this->gamma = Node["gamma"].as<type_real>();
    this->ispec = Node["ispec"].as<int>();
    this->medium_tag =
        specfem::element::from_string(Node["medium_tag"].as<std::string>());
  };

  type_real xi;
  type_real gamma;
  int ispec;
  specfem::element::medium_tag medium_tag;
};

struct solutions {
public:
  solutions() = default;
  solutions(const YAML::Node &Node) {
    if (Node["source"].IsDefined()) {
      auto source_node = Node["source"];
      this->source = source_solution(source_node);
    }
  }
  source_solution source;
};

// Base config struct
template <specfem::dimension::type DimensionType> struct config {
public:
  config() : nproc(1) {};
  virtual ~config() = default;

  int get_nproc() { return nproc; }

protected:
  int nproc;
};

// 2D config specialization
template <> struct config<specfem::dimension::type::dim2> {
public:
  config() : nproc(1), elastic_wave("P_SV"), electromagnetic_wave("TE") {};
  config(const YAML::Node &Node) {
    nproc = Node["nproc"].as<int>();
    if (Node["elastic_wave"].IsDefined())
      elastic_wave = Node["elastic_wave"].as<std::string>();
    else
      // Default to P_SV if not defined
      elastic_wave = "P_SV";

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

// 3D config specialization
template <> struct config<specfem::dimension::type::dim3> {
public:
  config() : nproc(1) {};
  config(const YAML::Node &Node) { nproc = Node["nproc"].as<int>(); }

  int get_nproc() { return nproc; }

private:
  int nproc;
};

template <specfem::dimension::type DimensionType> struct Test {
public:
  Test(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    suffix = Node["suffix"].as<std::string>();
    YAML::Node databases = Node["databases"];
    YAML::Node configuration = Node["config"];
    YAML::Node solutions_node = Node["solutions"];

    try {
      database = test_configuration::database<DimensionType>(databases);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Error in test configuration: " + name + "\n" +
                               e.what());
    }
    try {
      config = test_configuration::config<DimensionType>(configuration);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Error in test configuration: " + name + "\n" +
                               e.what());
    }

    try {
      solutions = test_configuration::solutions(solutions_node);
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

  std::string get_suffix() { return suffix; }

  std::string name;
  std::string description;
  std::string suffix;
  test_configuration::database<DimensionType> database;
  test_configuration::config<DimensionType> config;
  test_configuration::solutions solutions;
};

// 2D Test specialization with elastic wave methods
template <> struct Test<specfem::dimension::type::dim2> {
public:
  Test(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    suffix = Node["suffix"].as<std::string>();
    YAML::Node databases = Node["databases"];
    YAML::Node configuration = Node["config"];
    YAML::Node solutions_node = Node["solutions"];

    try {
      database = test_configuration::database<specfem::dimension::type::dim2>(
          databases);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Error in test configuration: " + name + "\n" +
                               e.what());
    }
    try {
      config = test_configuration::config<specfem::dimension::type::dim2>(
          configuration);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Error in test configuration: " + name + "\n" +
                               e.what());
    }

    try {
      solutions = test_configuration::solutions(solutions_node);
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
  test_configuration::database<specfem::dimension::type::dim2> database;
  test_configuration::config<specfem::dimension::type::dim2> config;
  test_configuration::solutions solutions;
};
} // namespace test_configuration

// ------------------------------------------------------------------------

template <specfem::dimension::type DimensionType>
class Assembly : public ::testing::Test {

protected:
  class Iterator {
  public:
    Iterator(
        test_configuration::Test<DimensionType> *p_Test,
        specfem::mesh::mesh<DimensionType> *p_mesh,
        std::vector<std::shared_ptr<specfem::sources::source<DimensionType> > >
            *p_sources,
        std::vector<
            std::shared_ptr<specfem::receivers::receiver<DimensionType> > >
            *p_stations,
        std::string *p_suffixes,
        specfem::assembly::assembly<DimensionType> *p_assembly)
        : p_Test(p_Test), p_mesh(p_mesh), p_sources(p_sources),
          p_stations(p_stations), p_suffixes(p_suffixes),
          p_assembly(p_assembly) {}

    std::tuple<
        test_configuration::Test<DimensionType>,
        specfem::mesh::mesh<DimensionType>,
        std::vector<std::shared_ptr<specfem::sources::source<DimensionType> > >,
        std::vector<
            std::shared_ptr<specfem::receivers::receiver<DimensionType> > >,
        std::string, specfem::assembly::assembly<DimensionType> >
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
    test_configuration::Test<DimensionType> *p_Test;
    specfem::mesh::mesh<DimensionType> *p_mesh;
    std::vector<std::shared_ptr<specfem::sources::source<DimensionType> > >
        *p_sources;
    std::vector<std::shared_ptr<specfem::receivers::receiver<DimensionType> > >
        *p_stations;
    std::string *p_suffixes;
    specfem::assembly::assembly<DimensionType> *p_assembly;
  };

  Assembly();

  Iterator begin() {
    return Iterator(&Tests[0], &Meshes[0], &Sources[0], &Stations[0],
                    &suffixes[0], &assemblies[0]);
  }

  Iterator end() {
    return Iterator(&Tests[Tests.size()], &Meshes[Meshes.size()],
                    &Sources[Sources.size()], &Stations[Stations.size()],
                    &suffixes[suffixes.size()], &assemblies[assemblies.size()]);
  }

  std::vector<test_configuration::Test<DimensionType> > Tests;
  std::vector<specfem::mesh::mesh<DimensionType> > Meshes;
  std::vector<
      std::vector<std::shared_ptr<specfem::sources::source<DimensionType> > > >
      Sources;
  std::vector<std::vector<
      std::shared_ptr<specfem::receivers::receiver<DimensionType> > > >
      Stations;
  std::vector<std::string> suffixes;
  std::vector<specfem::assembly::assembly<DimensionType> > assemblies;
};

// Template specializations
using Assembly2D = Assembly<specfem::dimension::type::dim2>;
using Assembly3D = Assembly<specfem::dimension::type::dim3>;
