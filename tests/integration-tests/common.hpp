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
  /// Directory of the reference seismograms; empty if the case has none
  std::string traces;
  /// How the seismograms are checked: fortran | analytic | serial |
  /// no-injection
  std::string compare = "fortran";
  /// Configuration of the injection plugin; empty if the case has no injection
  std::string plugin_config;

  static CaseConfig load(const Case &c);
};

/// Network and station name of one line of a STATIONS file
struct Station {
  std::string network;
  std::string name;

  /// `NETWORK.STATION`
  std::string full_name() const { return network + "." + name; }
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

/**
 * @brief Expect the velocity seismograms written to @p simulated to follow the
 * traces in @p reference.
 *
 * For every station and every component that is not negligible, the relative
 * L2 error must be within @p tolerance and the cross-correlation lag below a
 * quarter of @p dt.
 */
void expect_matches_reference_curves(const std::filesystem::path &simulated,
                                     const std::filesystem::path &reference,
                                     const std::vector<Station> &stations,
                                     const type_real dt, const double tolerance,
                                     const std::string &name);

/**
 * @brief Expect the velocity seismograms of two runs to be bit-identical, and
 * not zero.
 */
void expect_bitwise_equal(const std::filesystem::path &with,
                          const std::filesystem::path &without,
                          const std::vector<Station> &stations);

} // namespace integration_tests

namespace integration_tests_impl {

/// Euclidean norm
double norm(const std::vector<double> &v);

/// Lag of @p simulated behind @p reference, from the peak of their
/// cross-correlation refined by a parabola through its three samples
double lag(const std::vector<double> &simulated,
           const std::vector<double> &reference, const double dt);

/// Read `<station>.S3.?X<component>.semv` from a directory; the band code
/// depends on dt, so it is matched rather than spelled out.
specfem::integration_tests::Trace
find_velocity_trace(const std::filesystem::path &directory,
                    const specfem::integration_tests::Station &station,
                    const char component);

} // namespace integration_tests_impl
} // namespace specfem
