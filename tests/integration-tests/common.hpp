#pragma once

#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace specfem {
namespace integration_tests {

/**
 * @brief One simulation listed in configurations/tests.yaml.
 */
struct Case {
  std::string name;
  /// Case directory, relative to the test working directory
  std::string directory;
  int nproc = 1;
};

/// gtest prints a case by name rather than as bytes
void PrintTo(const Case &c, std::ostream *os);

/**
 * @brief Cases of the given sections of configurations/tests.yaml that run on
 * @p nproc MPI processes, sorted by name.
 *
 * A section name is also the directory under configurations/dim3/ that holds
 * the case.
 */
std::vector<Case> load_cases(const std::vector<std::string> &sections,
                             const int nproc);

/**
 * @brief Contents of a case's config.yaml.
 */
struct CaseConfig {
  int nproc = 1;
  double tolerance = 0;
  /// Parameter file of the simulation
  std::string specfem_config;
  /// Directory of the reference seismograms
  std::string traces;

  static CaseConfig load(const Case &c);
};

/// Network and station name of one line of a STATIONS file
struct Station {
  std::string network;
  std::string name;
};

/// Stations of a STATIONS file, in file order
std::vector<Station> read_stations(const std::string &stations_file);

/// Time and value columns of one seismogram file
struct Trace {
  std::vector<double> time;
  std::vector<double> value;
};

Trace read_trace(const std::filesystem::path &file);

/**
 * @brief Output directory of one run.
 *
 * Only the main rank touches the filesystem: a rank clearing the directory
 * after another has created its output would make the seismogram writer fail
 * on one rank and leave the others waiting at a barrier.
 */
class RunDirectory {
public:
  explicit RunDirectory(const std::string &name);

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

/**
 * @brief Expect the seismograms written to @p simulated to match the Fortran
 * reference traces of the case.
 *
 * Per station, the L2 error over every component and seismogram type is
 * compared with the reference norm. A reference that is zero everywhere must be
 * matched by an absolute error within the tolerance.
 */
void expect_matches_reference_traces(
    const std::filesystem::path &simulated, const std::string &reference,
    const std::vector<Station> &stations,
    const std::vector<specfem::enums::wavefield> &seismogram_types,
    const type_real dt, const double tolerance);

} // namespace integration_tests
} // namespace specfem
