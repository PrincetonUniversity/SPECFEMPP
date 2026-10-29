// End-to-end injection tests: run specfem 3d on the all-Stacey box with the
// plane-wave plugin and compare the velocity seismograms with stored
// references. The cases are listed in injection/tests.yaml.
//
// Built only when SPECFEM_INJECTION_TEST_PLUGIN names a built plane-wave
// plugin; SPECFEM_INJECTION_TEST_NPROC selects the cases for this executable's
// MPI size.

#include "SPECFEM_Environment.hpp"
#include "specfem/injection/method/registry.hpp"
#include "specfem/injection/options.hpp"
#include "specfem/io.hpp"
#include "specfem/mpi.hpp"
#include "specfem/program/dim3/program.hpp"
#include "yaml-cpp/yaml.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef SPECFEM_INJECTION_TEST_NPROC
#define SPECFEM_INJECTION_TEST_NPROC 1
#endif

namespace specfem::injection_e2e_test {

const std::filesystem::path fixture_dir = "injection";

struct Case {
  std::string name;
  std::string mesh;
  int nproc = 1;
  std::string plugin_config;
  std::string source;
  int nstep = 0;
  std::string compare;
  std::string reference;
  double tolerance = 0;
};

/// gtest prints the case by name rather than as bytes
void PrintTo(const Case &c, std::ostream *os) { *os << c.name; }

struct Fixture {
  double dt = 0;
  double t0 = 0;
  std::string quadrature;
  std::string stations;
  std::map<std::string, std::string> meshes;
  std::vector<Case> cases;
};

Fixture load_fixture() {
  const YAML::Node root = YAML::LoadFile((fixture_dir / "tests.yaml").string());
  Fixture fixture;
  fixture.dt = root["fixture"]["dt"].as<double>();
  fixture.t0 = root["fixture"]["t0"].as<double>();
  fixture.quadrature = root["fixture"]["quadrature"].as<std::string>();
  fixture.stations = root["fixture"]["stations"].as<std::string>();
  for (const auto &mesh : root["meshes"])
    fixture.meshes[mesh.first.as<std::string>()] =
        mesh.second.as<std::string>();
  for (const auto &node : root["cases"]) {
    Case c;
    c.name = node["name"].as<std::string>();
    c.mesh = node["mesh"] ? node["mesh"].as<std::string>() : "serial";
    c.nproc = node["nproc"] ? node["nproc"].as<int>() : 1;
    c.plugin_config = node["plugin-config"].as<std::string>();
    c.source = node["source"].as<std::string>();
    c.nstep = node["nstep"].as<int>();
    c.compare = node["compare"].as<std::string>();
    if (node["reference"])
      c.reference = node["reference"].as<std::string>();
    if (node["tolerance"])
      c.tolerance = node["tolerance"].as<double>();
    fixture.cases.push_back(c);
  }
  return fixture;
}

/// Station names (NET.STA) in file order; a line is `STA NET y x elevation z`
std::vector<std::string> station_names(const Fixture &fixture) {
  std::ifstream file(fixture_dir / fixture.stations);
  std::vector<std::string> names;
  std::string station, network, rest;
  while (file >> station >> network && std::getline(file, rest))
    names.push_back(network + "." + station);
  return names;
}

/// Time and value columns of one seismogram
struct Trace {
  std::vector<double> time;
  std::vector<double> value;
};

/// Read `<station>.S3.?X<component>.semv` from a directory; the band code
/// depends on dt, so it is matched rather than spelled out.
Trace read_trace(const std::filesystem::path &directory,
                 const std::string &station, const char component) {
  const std::string prefix = station + ".S3.";
  const std::string suffix = std::string("X") + component + ".semv";
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    const std::string name = entry.path().filename().string();
    if (name.size() == prefix.size() + 1 + suffix.size() &&
        name.starts_with(prefix) && name.ends_with(suffix)) {
      Trace trace;
      std::ifstream file(entry.path());
      double t, v;
      while (file >> t >> v) {
        trace.time.push_back(t);
        trace.value.push_back(v);
      }
      return trace;
    }
  }
  throw std::runtime_error("No seismogram for " + station + " component " +
                           component + " in " + directory.string());
}

/// Lag of @p simulated behind @p reference, from the peak of their
/// cross-correlation refined by a parabola through its three samples
double lag(const std::vector<double> &simulated,
           const std::vector<double> &reference, const double dt) {
  const int n = static_cast<int>(reference.size());
  auto correlation = [&](const int shift) {
    double sum = 0;
    for (int i = std::max(0, shift); i < std::min(n, n + shift); ++i)
      sum += simulated[i] * reference[i - shift];
    return sum;
  };
  int best = 0;
  double best_value = correlation(0);
  // Plane-wave lags are a few samples at most; search well beyond that.
  for (int shift = -50; shift <= 50; ++shift) {
    const double value = correlation(shift);
    if (value > best_value) {
      best_value = value;
      best = shift;
    }
  }
  const double before = correlation(best - 1);
  const double after = correlation(best + 1);
  const double curvature = before - 2 * best_value + after;
  const double offset = curvature != 0 ? 0.5 * (before - after) / curvature : 0;
  return (best + offset) * dt;
}

double norm(const std::vector<double> &v) {
  double sum = 0;
  for (const double x : v)
    sum += x * x;
  return std::sqrt(sum);
}

class RunDirectory {
public:
  // Only the main rank touches the filesystem: a rank clearing the directory
  // after another has created its output would make the seismogram writer fail
  // on one rank and leave the others waiting.
  explicit RunDirectory(const std::string &name)
      : path_(std::filesystem::absolute("injection_e2e_runs") / name) {
    if (specfem::MPI::main_proc()) {
      std::filesystem::remove_all(path_);
      std::filesystem::create_directories(path_);
    }
    specfem::MPI::sync_all();
  }

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

YAML::Node parameters(const Fixture &fixture, const Case &c,
                      const std::filesystem::path &output,
                      const bool with_injection) {
  const auto absolute = [](const std::filesystem::path &path) {
    return std::filesystem::absolute(fixture_dir / path).string();
  };
  YAML::Node root;
  YAML::Node p = root["parameters"];
  p["header"]["title"] = c.name;
  p["header"]["description"] = "end-to-end injection test";
  auto setup = p["simulation-setup"];
  setup["quadrature"]["quadrature-type"] = fixture.quadrature;
  auto scheme = setup["solver"]["time-marching"]["time-scheme"];
  scheme["type"] = "Newmark";
  scheme["dt"] = fixture.dt;
  scheme["nstep"] = c.nstep;
  auto seismogram = setup["simulation-mode"]["forward"]["writer"]["seismogram"];
  seismogram["format"] = "ascii";
  seismogram["directory"] = output.string();
  p["receivers"]["stations"] = absolute(fixture.stations);
  p["receivers"]["angle"] = 0.0;
  p["receivers"]["seismogram-type"].push_back("velocity");
  p["receivers"]["nstep_between_samples"] = 1;
  p["run-setup"]["number-of-processors"] = c.nproc;
  p["run-setup"]["number-of-runs"] = 1;
  p["databases"]["mesh-database"] = absolute(fixture.meshes.at(c.mesh));
  p["sources"] = absolute(c.source);
  if (with_injection)
    p["injection"]["method"] = "stacey";
  return root;
}

void run(const Fixture &fixture, const Case &c,
         const std::filesystem::path &output, const bool with_injection) {
  if (specfem::MPI::main_proc())
    std::filesystem::create_directories(output);
  specfem::MPI::sync_all();
  std::optional<specfem::injection::PluginOptions> plugin;
  if (with_injection)
    plugin = specfem::injection::PluginOptions{
      SPECFEM_INJECTION_TEST_PLUGIN,
      std::filesystem::absolute(fixture_dir / c.plugin_config).string()
    };
  specfem::program::program_3d(parameters(fixture, c, output, with_injection),
                               {}, false, plugin,
                               specfem::injection::default_method_registry());
  specfem::MPI::sync_all();
}

/// Ranks whose part of the mesh has no injection face
int ranks_without_injection_points(const Fixture &fixture, const Case &c) {
  const auto database = specfem::MPI::format_proc_filename(
      std::filesystem::absolute(fixture_dir / fixture.meshes.at(c.mesh))
          .string());
  const auto mesh =
      specfem::io::read_3d_mesh(database, specfem::attenuation::Setup{});
  int empty = mesh.boundaries.injection_surface.nfaces == 0 ? 1 : 0;
#ifdef SPECFEM_ENABLE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &empty, 1, MPI_INT, MPI_SUM,
                specfem::MPI::communicator());
#endif
  return empty;
}

void expect_bitwise_equal(const Fixture &fixture,
                          const std::filesystem::path &with,
                          const std::filesystem::path &without) {
  for (const auto &station : station_names(fixture))
    for (const char component : { 'X', 'Y', 'Z' }) {
      const Trace a = read_trace(with, station, component);
      const Trace b = read_trace(without, station, component);
      ASSERT_EQ(a.value.size(), b.value.size());
      int differing = 0;
      for (std::size_t i = 0; i < a.value.size(); ++i)
        differing += a.value[i] != b.value[i];
      EXPECT_EQ(differing, 0)
          << station << " " << component << ": " << differing
          << " samples differ from the run without injection";
      EXPECT_GT(norm(b.value), 0)
          << station << " " << component << ": the field must not be zero";
    }
}

void expect_matches_reference(const Fixture &fixture, const Case &c,
                              const std::filesystem::path &output) {
  const std::filesystem::path reference = fixture_dir / c.reference;
  double worst_error = 0, worst_lag = 0;
  for (const auto &station : station_names(fixture)) {
    std::array<Trace, 3> simulated, expected;
    double largest = 0;
    for (int i = 0; i < 3; ++i) {
      simulated[i] = read_trace(output, station, "XYZ"[i]);
      expected[i] = read_trace(reference, station, "XYZ"[i]);
      ASSERT_EQ(simulated[i].value.size(), expected[i].value.size())
          << station << ": trace lengths differ";
      ASSERT_NEAR(simulated[i].time.front(), expected[i].time.front(),
                  1e-6 * fixture.dt)
          << station << ": traces start at different times";
      largest = std::max(largest, norm(expected[i].value));
    }
    for (int i = 0; i < 3; ++i) {
      const auto &sim = simulated[i].value;
      const auto &ref = expected[i].value;
      std::vector<double> difference(sim.size());
      for (std::size_t s = 0; s < sim.size(); ++s)
        difference[s] = sim[s] - ref[s];
      // A component that barely moves is measured against the largest one.
      const double scale = std::max(norm(ref), 1e-2 * largest);
      const double error = norm(difference) / scale;
      worst_error = std::max(worst_error, error);
      EXPECT_LE(error, c.tolerance)
          << station << " " << "XYZ"[i] << ": relative L2 error " << error;
      if (norm(ref) > 1e-2 * largest) {
        const double l = lag(sim, ref, fixture.dt);
        worst_lag = std::max(worst_lag, std::abs(l));
        EXPECT_LT(std::abs(l), 0.25 * fixture.dt)
            << station << " " << "XYZ"[i] << ": lag " << l / fixture.dt
            << " dt";
      }
    }
  }
  std::cout << c.name << ": worst relative L2 error " << worst_error
            << " (tolerance " << c.tolerance << "), worst |lag| "
            << worst_lag / fixture.dt << " dt\n";
}

class InjectionEndToEnd : public ::testing::TestWithParam<Case> {};

TEST_P(InjectionEndToEnd, Run) {
  if (!SPECFEMEnvironment::IsMPISizeValid())
    GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();

  const Fixture fixture = load_fixture();
  const Case &c = GetParam();
  const RunDirectory directory(c.name);
  const auto with = directory.path() / "with_injection";
  run(fixture, c, with, true);

  const int empty = ranks_without_injection_points(fixture, c);
  if (specfem::MPI::main_proc())
    std::cout << c.name << ": " << empty << " of " << c.nproc
              << " ranks own no injection points\n";

  if (c.compare == "no-injection") {
    const auto without = directory.path() / "without_injection";
    run(fixture, c, without, false);
    if (specfem::MPI::main_proc())
      expect_bitwise_equal(fixture, with, without);
  } else if (specfem::MPI::main_proc()) {
    expect_matches_reference(fixture, c, with);
  }
}

std::vector<Case> cases_for_this_size() {
  std::vector<Case> selected;
  for (const auto &c : load_fixture().cases)
    if (c.nproc == SPECFEM_INJECTION_TEST_NPROC)
      selected.push_back(c);
  return selected;
}

// Instance names are the case names: CMake registers one CTest entry per case.
INSTANTIATE_TEST_SUITE_P(Cases, InjectionEndToEnd,
                         ::testing::ValuesIn(cases_for_this_size()),
                         [](const ::testing::TestParamInfo<Case> &info) {
                           return info.param.name;
                         });

} // namespace specfem::injection_e2e_test

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(
      new SPECFEMEnvironment(SPECFEM_INJECTION_TEST_NPROC));
  return RUN_ALL_TESTS();
}
