#include "common.hpp"
#include "specfem/io.hpp"
#include "specfem/mpi.hpp"
#include "yaml-cpp/yaml.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>

void specfem::integration_tests::PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

std::vector<specfem::integration_tests::Case>
specfem::integration_tests::load_cases(const std::vector<std::string> &sections,
                                       const int nproc) {
  const YAML::Node root = YAML::LoadFile("configurations/tests.yaml");
  std::vector<Case> cases;
  for (const auto &section : sections) {
    for (const auto &entry : root[section]) {
      if (entry.second.as<int>() != nproc)
        continue;
      Case c;
      c.name = entry.first.as<std::string>();
      c.directory = "configurations/dim3/" + section + "/" + c.name;
      c.nproc = nproc;
      cases.push_back(c);
    }
  }
  std::sort(cases.begin(), cases.end(),
            [](const Case &a, const Case &b) { return a.name < b.name; });
  return cases;
}

specfem::integration_tests::CaseConfig
specfem::integration_tests::CaseConfig::load(const Case &c) {
  const std::string file = c.directory + "/config.yaml";
  YAML::Node node;
  try {
    node = YAML::LoadFile(file);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load config file: " + file + " - " +
                             e.what());
  }
  CaseConfig config;
  config.nproc = node["config"]["nproc"].as<int>();
  config.tolerance = node["config"]["tolerance"].as<double>();
  config.specfem_config =
      c.directory + "/" + node["databases"]["specfem_config"].as<std::string>();
  const auto traces = node["databases"]["traces"].as<std::string>();
  if (!traces.empty())
    config.traces = c.directory + "/" + traces;
  if (node["compare"])
    config.compare = node["compare"].as<std::string>();
  if (node["injection"])
    config.plugin_config = c.directory + "/" +
                           node["injection"]["plugin-config"].as<std::string>();
  return config;
}

std::vector<specfem::integration_tests::Station>
specfem::integration_tests::read_stations(const std::string &stations_file) {
  std::ifstream file(stations_file);
  if (!file)
    throw std::runtime_error("Cannot open stations file " + stations_file);
  std::vector<Station> stations;
  // A line is `STATION NETWORK latitude longitude elevation burial`
  std::string name, network, rest;
  while (file >> name >> network && std::getline(file, rest))
    stations.push_back({ network, name });
  return stations;
}

specfem::integration_tests::Trace
specfem::integration_tests::read_trace(const std::filesystem::path &file) {
  std::ifstream stream(file);
  if (!stream)
    throw std::runtime_error("Cannot open seismogram " + file.string());
  Trace trace;
  double time, value;
  while (stream >> time >> value) {
    trace.time.push_back(time);
    trace.value.push_back(value);
  }
  return trace;
}

specfem::integration_tests::RunDirectory::RunDirectory(const std::string &name)
    : path_(std::filesystem::absolute("integration_test_runs") / name) {
  if (specfem::MPI::main_proc()) {
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }
  specfem::MPI::sync_all();
}

void specfem::integration_tests::expect_matches_reference_traces(
    const std::filesystem::path &simulated, const std::string &reference,
    const std::vector<Station> &stations,
    const std::vector<specfem::enums::wavefield> &seismogram_types,
    const type_real dt, const double tolerance) {
  specfem::io::impl::ChannelGenerator channel_generator(dt);

  for (const auto &station : stations) {
    double error = 0.0;
    double reference_norm = 0.0;

    for (const auto seismogram_type : seismogram_types) {
      for (const auto &filename : channel_generator.get_station_filenames(
               station.network, station.name, "S3", seismogram_type)) {
        const Trace computed = read_trace(simulated / filename);
        const Trace expected =
            read_trace(std::filesystem::path(reference) / filename);

        ASSERT_LE(computed.value.size(), expected.value.size())
            << station.network << "." << station.name << " " << filename
            << ": fewer reference samples than simulated samples";

        for (std::size_t step = 0; step < computed.value.size(); ++step) {
          ASSERT_LE(std::abs(computed.time[step] - expected.time[step]), 1e-3)
              << filename << ": times do not match at sample " << step;
          const double difference = computed.value[step] - expected.value[step];
          error += difference * difference;
          reference_norm += expected.value[step] * expected.value[step];
        }
      }
    }

    error = std::sqrt(error);
    reference_norm = std::sqrt(reference_norm);

    // When the reference norm is zero both the reference and computed
    // seismograms must be zero for the test to pass.
    const bool failed = (reference_norm == 0)
                            ? (error > tolerance)
                            : (error / reference_norm > tolerance ||
                               std::isnan(error / reference_norm));
    EXPECT_FALSE(failed) << station.network << "." << station.name << ": error "
                         << error << " relative to norm " << reference_norm
                         << " exceeds tolerance " << tolerance;
  }
}

double specfem::integration_tests_impl::norm(const std::vector<double> &v) {
  double sum = 0;
  for (const double x : v)
    sum += x * x;
  return std::sqrt(sum);
}

double
specfem::integration_tests_impl::lag(const std::vector<double> &simulated,
                                     const std::vector<double> &reference,
                                     const double dt) {
  const int n = static_cast<int>(reference.size());
  const auto correlation = [&](const int shift) {
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

specfem::integration_tests::Trace
specfem::integration_tests_impl::find_velocity_trace(
    const std::filesystem::path &directory,
    const specfem::integration_tests::Station &station, const char component) {
  const std::string prefix = station.full_name() + ".S3.";
  const std::string suffix = std::string("X") + component + ".semv";
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    const std::string name = entry.path().filename().string();
    if (name.size() == prefix.size() + 1 + suffix.size() &&
        name.starts_with(prefix) && name.ends_with(suffix))
      return specfem::integration_tests::read_trace(entry.path());
  }
  throw std::runtime_error("No seismogram for " + station.full_name() +
                           " component " + component + " in " +
                           directory.string());
}

void specfem::integration_tests::expect_matches_reference_curves(
    const std::filesystem::path &simulated,
    const std::filesystem::path &reference,
    const std::vector<Station> &stations, const type_real dt,
    const double tolerance, const std::string &name) {
  double worst_error = 0, worst_lag = 0;
  for (const auto &station : stations) {
    std::array<Trace, 3> computed, expected;
    double largest = 0;
    for (int i = 0; i < 3; ++i) {
      computed[i] = integration_tests_impl::find_velocity_trace(
          simulated, station, "XYZ"[i]);
      expected[i] = integration_tests_impl::find_velocity_trace(
          reference, station, "XYZ"[i]);
      ASSERT_EQ(computed[i].value.size(), expected[i].value.size())
          << station.full_name() << ": trace lengths differ";
      ASSERT_NEAR(computed[i].time.front(), expected[i].time.front(), 1e-6 * dt)
          << station.full_name() << ": traces start at different times";
      largest =
          std::max(largest, integration_tests_impl::norm(expected[i].value));
    }
    for (int i = 0; i < 3; ++i) {
      const auto &sim = computed[i].value;
      const auto &ref = expected[i].value;
      std::vector<double> difference(sim.size());
      for (std::size_t s = 0; s < sim.size(); ++s)
        difference[s] = sim[s] - ref[s];
      // A component that barely moves is measured against the largest one.
      const double reference_norm = integration_tests_impl::norm(ref);
      const double scale = std::max(reference_norm, 1e-2 * largest);
      const double error = integration_tests_impl::norm(difference) / scale;
      worst_error = std::max(worst_error, error);
      EXPECT_LE(error, tolerance) << station.full_name() << " " << "XYZ"[i]
                                  << ": relative L2 error " << error;
      if (reference_norm > 1e-2 * largest) {
        const double l = integration_tests_impl::lag(sim, ref, dt);
        worst_lag = std::max(worst_lag, std::abs(l));
        EXPECT_LT(std::abs(l), 0.25 * dt)
            << station.full_name() << " " << "XYZ"[i] << ": lag " << l / dt
            << " dt";
      }
    }
  }
  std::cout << name << ": worst relative L2 error " << worst_error
            << " (tolerance " << tolerance << "), worst |lag| "
            << worst_lag / dt << " dt\n";
}

void specfem::integration_tests::expect_bitwise_equal(
    const std::filesystem::path &with, const std::filesystem::path &without,
    const std::vector<Station> &stations) {
  for (const auto &station : stations)
    for (const char component : { 'X', 'Y', 'Z' }) {
      const Trace a =
          integration_tests_impl::find_velocity_trace(with, station, component);
      const Trace b = integration_tests_impl::find_velocity_trace(
          without, station, component);
      ASSERT_EQ(a.value.size(), b.value.size());
      int differing = 0;
      for (std::size_t i = 0; i < a.value.size(); ++i)
        differing += a.value[i] != b.value[i];
      EXPECT_EQ(differing, 0)
          << station.full_name() << " " << component << ": " << differing
          << " samples differ from the run without injection";
      EXPECT_GT(integration_tests_impl::norm(b.value), 0)
          << station.full_name() << " " << component
          << ": the field must not be zero";
    }
}
