// Golden-comparison test: depth-grouped FK driver vs FKG1 reference files.
//
// For each scenario {E1,E2,E3,E4,A1}:
//   1. Read the FKG1 binary oracle.
//   2. Run run_fk_driver() twice: legacy stretch (asserted) and exact.
//   3. Parity gate: legacy run must match golden to 2e-5 at every (point,comp).
//   4. Print exact-mode improvement without asserting it.

#include "fk_golden_scenarios.hpp"
#include "specfem/injection/fk/impl/driver.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Implementation helpers
// ---------------------------------------------------------------------------
namespace driver_golden_tests_impl {

// FKG1 fixed-header layout (little-endian).
struct Fkg1Header {
  char magic[4];   // "FKG1"
  int32_t version; // 1
  int32_t num_points;
  int32_t num_components; // 7
  int32_t num_steps;
  int32_t np_resamp;
  int32_t nf_stored;
  int32_t nfft;
  double dt_sem;
  double dt_min;
  double origin_time;
  double reference_time;
};

// Per-point metadata record.
struct Fkg1Point {
  int32_t medium; // 0=elastic, 1=acoustic
  double tau;
  double z;
  double x;
  double y;
};

// Read a plain little-endian POD value from stream.
template <typename T> static bool read_pod(std::ifstream &fs, T &val) {
  fs.read(reinterpret_cast<char *>(&val), sizeof(T));
  return static_cast<bool>(fs);
}

// Read the full FKG1 file.  Returns false on any parse failure.
static bool read_fkg1(const std::string &path, Fkg1Header &hdr,
                      std::vector<Fkg1Point> &points,
                      std::vector<double> &values) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs.is_open()) {
    return false;
  }

  // Fixed header
  fs.read(hdr.magic, 4);
  if (!fs || hdr.magic[0] != 'F' || hdr.magic[1] != 'K' ||
      hdr.magic[2] != 'G' || hdr.magic[3] != '1') {
    return false;
  }
  if (!read_pod(fs, hdr.version))
    return false;
  if (!read_pod(fs, hdr.num_points))
    return false;
  if (!read_pod(fs, hdr.num_components))
    return false;
  if (!read_pod(fs, hdr.num_steps))
    return false;
  if (!read_pod(fs, hdr.np_resamp))
    return false;
  if (!read_pod(fs, hdr.nf_stored))
    return false;
  if (!read_pod(fs, hdr.nfft))
    return false;
  if (!read_pod(fs, hdr.dt_sem))
    return false;
  if (!read_pod(fs, hdr.dt_min))
    return false;
  if (!read_pod(fs, hdr.origin_time))
    return false;
  if (!read_pod(fs, hdr.reference_time))
    return false;

  // Per-point records
  points.resize(hdr.num_points);
  for (int pt = 0; pt < hdr.num_points; ++pt) {
    if (!read_pod(fs, points[pt].medium))
      return false;
    if (!read_pod(fs, points[pt].tau))
      return false;
    if (!read_pod(fs, points[pt].z))
      return false;
    if (!read_pod(fs, points[pt].x))
      return false;
    if (!read_pod(fs, points[pt].y))
      return false;
  }

  // Value block: double[num_points][num_components][num_steps]
  const std::size_t total = static_cast<std::size_t>(hdr.num_points) *
                            static_cast<std::size_t>(hdr.num_components) *
                            static_cast<std::size_t>(hdr.num_steps);
  values.resize(total);
  fs.read(reinterpret_cast<char *>(values.data()),
          static_cast<std::streamsize>(total * sizeof(double)));
  return static_cast<bool>(fs);
}

// g(point, comp, step) accessor into C-order flat array.
inline double g(const std::vector<double> &v, int np, int nc, int ns, int pt,
                int c, int n) {
  return v[static_cast<std::size_t>(pt) * static_cast<std::size_t>(nc * ns) +
           static_cast<std::size_t>(c) * static_cast<std::size_t>(ns) +
           static_cast<std::size_t>(n)];
}

// Find scenario by id.
static specfem::testing::fk_golden::GoldenScenario
find_scenario(const std::string &id) {
  for (const auto &sc : specfem::testing::fk_golden::all_scenarios()) {
    if (sc.id == id) {
      return sc;
    }
  }
  throw std::runtime_error("unknown scenario id: " + id);
}

// Compute per-(point,comp) relative error over valid steps.
// Returns err/scale; sets err_abs to the raw max absolute error.
static double compute_relative_error(
    const Kokkos::View<type_real ***, Kokkos::HostSpace> &out,
    const std::vector<double> &golden_vals, int num_points, int num_components,
    int num_steps, int np_resamp, int nf_stored, double peak, int pt, int c,
    double &err_abs) {
  auto valid_step = [&](int n) -> bool {
    const double ratio =
        static_cast<double>(n) / static_cast<double>(np_resamp);
    return ratio >= 24.0 && ratio <= static_cast<double>(nf_stored) - 12.0;
  };

  double err = 0.0;
  double scale_golden = 0.0;

  for (int n = 0; n < num_steps; ++n) {
    if (!valid_step(n))
      continue;
    const double gv = std::fabs(
        g(golden_vals, num_points, num_components, num_steps, pt, c, n));
    if (gv > scale_golden)
      scale_golden = gv;
    const double nv = std::fabs(
        static_cast<double>(out(pt, c, n)) -
        g(golden_vals, num_points, num_components, num_steps, pt, c, n));
    if (nv > err)
      err = nv;
  }

  err_abs = err;
  const double scale =
      (scale_golden > 1e-6 * peak) ? scale_golden : 1e-6 * peak;
  return err / scale;
}

} // namespace driver_golden_tests_impl

// ---------------------------------------------------------------------------
// Parameterized test fixture
// ---------------------------------------------------------------------------

class FkGolden : public ::testing::TestWithParam<std::string> {};

TEST_P(FkGolden, MatchesReference) {
  const std::string id = GetParam();

  // 1. Load the golden scenario inputs.
  const specfem::testing::fk_golden::GoldenScenario sc =
      driver_golden_tests_impl::find_scenario(id);

  // 2. Read the FKG1 binary oracle.
  const std::string path =
      std::string("data/injection/fk_golden/") + id + ".bin";
  driver_golden_tests_impl::Fkg1Header hdr{};
  std::vector<driver_golden_tests_impl::Fkg1Point> pts;
  std::vector<double> golden_vals;

  ASSERT_TRUE(driver_golden_tests_impl::read_fkg1(path, hdr, pts, golden_vals))
      << "failed to read FKG1 file: " << path;

  const int num_points = hdr.num_points;
  const int num_components = hdr.num_components;
  const int num_steps = hdr.num_steps;
  const int np_resamp = hdr.np_resamp;
  const int nf_stored = hdr.nf_stored;

  // 3. Structural assertions.
  ASSERT_EQ(num_points, static_cast<int>(sc.points.x.extent(0)));
  ASSERT_EQ(num_steps, sc.window.nstep);
  ASSERT_EQ(nf_stored, 512);

  // Scenario peak over ALL (point, comp, step).
  double peak = 1e-30;
  for (int pt = 0; pt < num_points; ++pt) {
    for (int c = 0; c < num_components; ++c) {
      for (int n = 0; n < num_steps; ++n) {
        const double v = std::fabs(driver_golden_tests_impl::g(
            golden_vals, num_points, num_components, num_steps, pt, c, n));
        if (v > peak)
          peak = v;
      }
    }
  }

  // Helper: sample an FkResult into a host mirror view.
  auto sample_to_host = [&](const specfem::injection::fk::FkResult &result)
      -> Kokkos::View<type_real ***, Kokkos::HostSpace> {
    Kokkos::View<type_real ***> out_d("out", num_points, 7, num_steps);
    result.sample(0, num_steps, out_d);
    auto out_h = Kokkos::create_mirror_view(out_d);
    Kokkos::deep_copy(out_h, out_d);
    return out_h;
  };

  // 4. Run with legacy stretch (reproduce pre-fix reference exactly).
  const auto legacy_result = specfem::injection::fk_impl::run_fk_driver(
      sc.model, sc.wave, sc.window, sc.points, sc.compute_traction,
      sc.derivative, /*reproduce_specfem3d_reference=*/true);
  const auto legacy_out = sample_to_host(legacy_result);

  // 5. Run with physically-exact delay (the shipping path).
  const auto exact_result = specfem::injection::fk_impl::run_fk_driver(
      sc.model, sc.wave, sc.window, sc.points, sc.compute_traction,
      sc.derivative, /*reproduce_specfem3d_reference=*/false);
  const auto exact_out = sample_to_host(exact_result);

  // 6. Parity gate: legacy must match golden to 2e-5 everywhere.
  //    Exact-mode errors vs golden are reported but not asserted.
  double parity_max_e = 0.0;
  double exact_tau0_max_e = 0.0;
  double exact_tauNE_max_e = 0.0;

  for (int pt = 0; pt < num_points; ++pt) {
    const double tau_p = pts[pt].tau;
    const bool is_tau0 = (std::fabs(tau_p) < 1e-12);

    for (int c = 0; c < num_components; ++c) {
      // --- legacy parity gate ---
      double err_abs_legacy = 0.0;
      const double e_legacy = driver_golden_tests_impl::compute_relative_error(
          legacy_out, golden_vals, num_points, num_components, num_steps,
          np_resamp, nf_stored, peak, pt, c, err_abs_legacy);

      if (err_abs_legacy < 1e-9 * peak)
        continue;

      if (e_legacy > parity_max_e)
        parity_max_e = e_legacy;

      EXPECT_LE(e_legacy, 2e-5)
          << "parity gate exceeded for scenario " << id << " point=" << pt
          << " comp=" << c << " e=" << e_legacy;

      // --- exact-mode error (no assert) ---
      double err_abs_exact = 0.0;
      const double e_exact = driver_golden_tests_impl::compute_relative_error(
          exact_out, golden_vals, num_points, num_components, num_steps,
          np_resamp, nf_stored, peak, pt, c, err_abs_exact);

      if (err_abs_exact < 1e-9 * peak)
        continue;

      if (is_tau0) {
        if (e_exact > exact_tau0_max_e)
          exact_tau0_max_e = e_exact;
      } else {
        if (e_exact > exact_tauNE_max_e)
          exact_tauNE_max_e = e_exact;
      }
    }
  }

  std::printf("[golden] %s: parity_max_e(legacy)=%.3e"
              " exact_tau0_max_e=%.3e exact_tauNE_max_e=%.3e\n",
              id.c_str(), parity_max_e, exact_tau0_max_e, exact_tauNE_max_e);
}

INSTANTIATE_TEST_SUITE_P(Scenarios, FkGolden,
                         ::testing::Values("E1", "E2", "E3", "E4", "A1"));
