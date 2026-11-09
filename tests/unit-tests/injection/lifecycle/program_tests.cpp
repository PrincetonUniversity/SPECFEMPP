// End-to-end tests of the injection lifecycle in a forward 3D run: the program
// builds the method from a registry, creates and sets up the plugin, and the
// solver prepares and applies the method every step.

#include "../method/constant_force.hpp"
#include "../session/mock_plugin.hpp"
#include "recording_method.hpp"
#include "specfem/injection/method/registry.hpp"
#include "specfem/program/dim3/program.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace specfem::injection_lifecycle_test {

using specfem::injection_session_test::mock_calls;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

constexpr int nstep = 10;

// A private temporary directory with the source and station files of a run,
// removed with its contents.
class RunDirectory {
public:
  RunDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "specfempp_lifecycle_XXXXXX")
            .string();
    if (::mkdtemp(pattern.data()) == nullptr)
      throw std::runtime_error("mkdtemp failed");
    path_ = pattern;
    std::ofstream(path_ / "force.yaml") << R"(number-of-sources: 1
sources:
  - force:
      x : 50000.0
      y : 40000.0
      z : -30000.0
      source_surf: false
      fx : 0.0
      fy : 0.0
      fz : 1e10
      Ricker:
        factor: 1.0
        tshift: 0.0
        f0: 0.05
)";
    std::ofstream(path_ / "STATIONS")
        << "STA TEST 30000.0 40000.0 0.0 -20000.0\n";
  }
  ~RunDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  RunDirectory(const RunDirectory &) = delete;
  RunDirectory &operator=(const RunDirectory &) = delete;

  // Parameter file of a run on a test database, with an injection section.
  YAML::Node parameters(const std::string &database,
                        const std::string &simulation_mode,
                        const std::string &injection) const {
    const auto mesh =
        std::filesystem::absolute("data/dim3/" + database + "/database.bin");
    return YAML::Load(R"(parameters:
  header:
    title: injection lifecycle
    description: injection lifecycle test
  simulation-setup:
    quadrature:
      quadrature-type: GLL4
    solver:
      time-marching:
        time-scheme:
          type: Newmark
          dt: 0.1
          nstep: )" + std::to_string(nstep) +
                      R"(
    simulation-mode:
)" + simulation_mode +
                      R"(
  receivers:
    stations: ")" + (path_ / "STATIONS").string() +
                      R"("
    angle: 0.0
    seismogram-type:
      - displacement
    nstep_between_samples: 1
  run-setup:
    number-of-processors: 1
    number-of-runs: 1
  databases:
    mesh-database: ")" +
                      mesh.string() +
                      R"("
  sources: ")" + (path_ / "force.yaml").string() +
                      R"("
  injection: )" + injection +
                      "\n");
  }

  std::string forward() const {
    return "      forward:\n        writer:\n          seismogram:\n"
           "            format: ascii\n            directory: \"" +
           path_.string() + "\"";
  }

  std::string combined() const {
    return "      combined:\n        reader:\n          wavefield:\n"
           "            format: NPY\n            directory: \"" +
           path_.string() + "\"";
  }

private:
  std::filesystem::path path_;
};

specfem::injection::MethodRegistry test_registry() {
  specfem::injection::MethodRegistry registry;
  registry.add<RecordingMethod>();
  registry.add<specfem::injection::PointwiseSurfaceMethod<
      specfem::injection_method_test::ConstantForce>>();
  return registry;
}

specfem::injection::PluginOptions mock_plugin() {
  return { specfem::injection_session_test::mock_plugin_spec(), "ok" };
}

void run(const YAML::Node &parameters,
         const specfem::injection::MethodRegistry &registry) {
  specfem::program::program_3d(parameters, {}, false, mock_plugin(), registry);
}

std::string run_error(const YAML::Node &parameters) {
  try {
    run(parameters, test_registry());
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "<no error>";
}

TEST(InjectionLifecycle, ForwardRunDrivesPluginEveryStep) {
  specfem::injection_session_test::reset_mock_calls();
  method_record() = MethodRecord{};
  const RunDirectory directory;
  run(directory.parameters("EightNodeElasticInjectionStacey",
                           directory.forward(),
                           "{method: recording, window-steps: 4}"),
      test_registry());

  // The plugin was created, set up with one group per material system, and
  // destroyed when the run ended.
  EXPECT_EQ(mock_calls().create, 1);
  EXPECT_EQ(mock_calls().setup, 1);
  EXPECT_EQ(mock_calls().destroy, 1);
  EXPECT_GT(mock_calls().setup_points, 0);
  EXPECT_THAT(mock_calls().fill_starts, ElementsAre(0, 4, 8));

  std::vector<int> steps(nstep);
  std::iota(steps.begin(), steps.end(), 0);
  EXPECT_EQ(method_record().setups, 1);
  EXPECT_EQ(method_record().window_steps, 4);
  EXPECT_THAT(method_record().prepared_steps, ElementsAreArray(steps));
  EXPECT_THAT(method_record().elastic_applies, ElementsAreArray(steps));
}

TEST(InjectionLifecycle, PointwiseMethodRunsInTheSolver) {
  specfem::injection_session_test::reset_mock_calls();
  const RunDirectory directory;
  run(directory.parameters("EightNodeElasticInjectionStacey",
                           directory.forward(),
                           "{method: constant-force, window-steps: 4}"),
      test_registry());

  EXPECT_EQ(mock_calls().destroy, 1);
  EXPECT_THAT(mock_calls().fill_starts, ElementsAre(0, 4, 8));
  ASSERT_FALSE(mock_calls().groups.empty());
  const auto &elastic = mock_calls().groups.front();
  EXPECT_EQ(elastic.medium, specfem_medium_elastic);
  ASSERT_EQ(elastic.slots.size(), 2u);
  EXPECT_EQ(elastic.slots[0].quantity, specfem_quantity_displacement);
  EXPECT_EQ(elastic.slots[1].quantity, specfem_quantity_velocity);
}

TEST(InjectionLifecycle, CombinedRunIsRejectedBeforeTheMeshRead) {
  const RunDirectory directory;
  EXPECT_THAT(run_error(directory.parameters("EightNodeElasticInjectionStacey",
                                             directory.combined(),
                                             "{method: recording}")),
              HasSubstr("supported for forward simulations only"));
}

TEST(InjectionLifecycle, UnknownMethodIsRejectedBeforeTheMeshRead) {
  const RunDirectory directory;
  EXPECT_THAT(
      run_error(directory.parameters("EightNodeElasticInjectionStacey",
                                     directory.forward(), "{method: stacey}")),
      HasSubstr("Unknown injection method 'stacey'. Registered "
                "methods: constant-force recording."));
}

TEST(InjectionLifecycle, MeshWithoutInjectionFacesIsRejected) {
  const RunDirectory directory;
  EXPECT_THAT(
      run_error(directory.parameters("EightNodeElastic", directory.forward(),
                                     "{method: recording}")),
      HasSubstr("the mesh declares no injection faces"));
}

} // namespace specfem::injection_lifecycle_test
