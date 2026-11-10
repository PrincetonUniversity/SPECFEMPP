// Tests of the plane-wave plugin: the analytic field against finite
// differences, and the plugin's setup and fill through synthetic SDK structs.
// Plain C++: each CHECK counts a failure, and main returns non-zero if any
// check failed.

#include "config.hpp"
#include "plane_wave/plane_wave.hpp"
#include "plane_wave_plugin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace plane_wave_test {

int failures = 0;

#define CHECK(condition, message)                                              \
  do {                                                                         \
    if (!(condition)) {                                                        \
      ++plane_wave_test::failures;                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << ": " << message << "\n";     \
    }                                                                          \
  } while (false)

using plane_wave::Vector;

double norm(const Vector &v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

double dot(const Vector &a, const Vector &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

bool near(const Vector &a, const Vector &b, const double tolerance) {
  for (int i = 0; i < 3; ++i)
    if (std::abs(a[i] - b[i]) > tolerance)
      return false;
  return true;
}

const plane_wave::Material granite{ 2700, 6000, 3464 };

plane_wave::PlaneWave make_wave(const plane_wave::WaveType type,
                                const plane_wave::SourceTimeFunction &stf) {
  return { granite, type, 30, 45, { 100, -200, -1000 }, 0.1, 1e-3, stf };
}

// Times at which the wavefront is near point x, so the field is not
// negligible there.
std::vector<double> times_near(const plane_wave::PlaneWave &wave,
                               const Vector &x) {
  const Vector origin{ 100, -200, -1000 };
  double distance = 0;
  for (int i = 0; i < 3; ++i)
    distance += wave.direction()[i] * (x[i] - origin[i]);
  const double arrival = 0.1 + distance / wave.speed();
  return { arrival - 0.2, arrival - 0.05, arrival, arrival + 0.07 };
}

void test_directions() {
  const auto stf = plane_wave::SourceTimeFunction::ricker(2);
  const plane_wave::PlaneWave up(granite, plane_wave::WaveType::p, 0, 0,
                                 { 0, 0, 0 }, 0, 1, stf);
  CHECK(near(up.direction(), { 0, 0, 1 }, 1e-12),
        "take-off 0 must point along +z");
  const plane_wave::PlaneWave y(granite, plane_wave::WaveType::p, 90, 90,
                                { 0, 0, 0 }, 0, 1, stf);
  CHECK(near(y.direction(), { 0, 1, 0 }, 1e-12),
        "take-off 90, azimuth 90 must point along +y");

  const auto p = make_wave(plane_wave::WaveType::p, stf);
  CHECK(near(p.polarization(), p.direction(), 1e-12),
        "P must be polarized along its direction");
  CHECK(std::abs(p.speed() - granite.vp) < 1e-12, "P must travel at vp");

  const auto sv = make_wave(plane_wave::WaveType::sv, stf);
  const Vector &k = sv.direction();
  const Vector &d = sv.polarization();
  CHECK(std::abs(dot(d, k)) < 1e-12, "SV must be perpendicular to k");
  CHECK(std::abs(norm(d) - 1) < 1e-12, "SV polarization must be a unit vector");
  // In the vertical plane through k: perpendicular to the horizontal normal
  // of that plane.
  const Vector horizontal{ -k[1], k[0], 0 };
  CHECK(std::abs(dot(d, horizontal)) < 1e-12,
        "SV must lie in the vertical plane of k");
  CHECK(std::abs(sv.speed() - granite.vs) < 1e-12, "SV must travel at vs");
}

void test_time_derivatives(const plane_wave::PlaneWave &wave,
                           const std::string &name) {
  const Vector x{ 300, 150, -400 };
  const double h = 1e-5;
  for (const double t : times_near(wave, x)) {
    const Vector v = wave.velocity(x, t);
    const Vector a = wave.acceleration(x, t);
    Vector dudt, dvdt;
    const Vector u_plus = wave.displacement(x, t + h);
    const Vector u_minus = wave.displacement(x, t - h);
    const Vector v_plus = wave.velocity(x, t + h);
    const Vector v_minus = wave.velocity(x, t - h);
    for (int i = 0; i < 3; ++i) {
      dudt[i] = (u_plus[i] - u_minus[i]) / (2 * h);
      dvdt[i] = (v_plus[i] - v_minus[i]) / (2 * h);
    }
    CHECK(near(v, dudt, 1e-6 * std::max(1.0, norm(v)) + 1e-9),
          name << ": velocity must be du/dt at t = " << t);
    CHECK(near(a, dvdt, 1e-6 * std::max(1.0, norm(a)) + 1e-8),
          name << ": acceleration must be dv/dt at t = " << t);
  }
}

// Traction from sigma(grad u) . n with a finite-difference gradient, against
// the analytic traction.
void test_traction(const plane_wave::PlaneWave &wave, const std::string &name) {
  const Vector x{ -250, 400, -600 };
  const Vector n{ 0.48, -0.6, 0.64 };
  const double h = 1e-2;
  const double lambda = granite.lambda();
  const double mu = granite.mu();
  for (const double t : times_near(wave, x)) {
    double grad[3][3];
    for (int j = 0; j < 3; ++j) {
      Vector plus = x, minus = x;
      plus[j] += h;
      minus[j] -= h;
      const Vector u_plus = wave.displacement(plus, t);
      const Vector u_minus = wave.displacement(minus, t);
      for (int i = 0; i < 3; ++i)
        grad[i][j] = (u_plus[i] - u_minus[i]) / (2 * h);
    }
    const double divergence = grad[0][0] + grad[1][1] + grad[2][2];
    Vector expected{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        const double sigma =
            mu * (grad[i][j] + grad[j][i]) + (i == j ? lambda * divergence : 0);
        expected[i] += sigma * n[j];
      }
    const Vector traction = wave.traction(x, n, t);
    CHECK(near(traction, expected, 1e-6 * std::max(1.0, norm(expected))),
          name << ": traction must be sigma(grad u) n at t = " << t);
  }
}

void test_sv_is_divergence_free() {
  const auto sv = make_wave(plane_wave::WaveType::sv,
                            plane_wave::SourceTimeFunction::gaussian(0.1));
  const Vector x{ 50, 60, -70 };
  for (const double t : times_near(sv, x)) {
    const auto epsilon = sv.strain(x, t);
    const double scale = std::abs(epsilon[0][2]) + std::abs(epsilon[1][2]) +
                         std::abs(epsilon[0][1]) + 1e-30;
    CHECK(std::abs(epsilon[0][0] + epsilon[1][1] + epsilon[2][2]) <
              1e-12 * scale,
          "SV strain must have zero trace at t = " << t);
  }
}

plane_wave::Config make_config() {
  return plane_wave::parse_config(R"(
material: {rho: 2700, vp: 6000, vs: 3464}
wave:
  type: SV
  take-off: 30
  azimuth: 45
  origin: [100, -200, -1000]
  origin-time: 0.1
  amplitude: 1.0e-3
source-time-function: {type: ricker, f0: 2.0}
material-tolerance: 1.0e-6
)");
}

specfem::plugin::SimulationInfo make_info(const int dimension = 3) {
  specfem_simulation_info info{};
  info.struct_size = sizeof(info);
  info.dimension = dimension;
  info.nstep = 10;
  info.t0 = -0.2;
  info.dt = 0.05;
  return specfem::plugin::SimulationInfo(info);
}

/// Synthetic injection points: an elastic isotropic group requesting velocity
/// and traction, followed by an empty acoustic group.
class Scene {
public:
  explicit Scene(const plane_wave::Material &material) {
    for (int p = 0; p < npoints; ++p) {
      x_.push_back(-300 + 120.0 * p);
      y_.push_back(80.0 * p);
      z_.push_back(-500 + 50.0 * p);
      const Vector n{ 0.36 * (p % 2 ? 1 : -1), 0.48, 0.8 };
      nx_.push_back(n[0]);
      ny_.push_back(n[1]);
      nz_.push_back(n[2]);
      const double mu = material.rho * material.vs * material.vs;
      properties_.push_back(material.rho * material.vp * material.vp -
                            4 * mu / 3);
      properties_.push_back(mu);
      properties_.push_back(material.rho);
    }

    elastic_.struct_size = sizeof(elastic_);
    elastic_.medium = specfem_medium_elastic;
    elastic_.property = specfem_property_isotropic;
    elastic_.nproperties = 3;
    elastic_.begin = 0;
    elastic_.count = npoints;
    elastic_.property_names = names_;
    elastic_.properties = properties_.data();
    elastic_.ncomponents = ncomponents;
    elastic_.nslots = 2;
    elastic_.slots = slots_;

    acoustic_.struct_size = sizeof(acoustic_);
    acoustic_.medium = specfem_medium_acoustic;
    acoustic_.property = specfem_property_isotropic;
    acoustic_.begin = npoints;

    points_.struct_size = sizeof(points_);
    points_.ngroups = 2;
    points_.n = npoints;
    points_.x = x_.data();
    points_.y = y_.data();
    points_.z = z_.data();
    points_.nx = nx_.data();
    points_.ny = ny_.data();
    points_.nz = nz_.data();
    points_.groups = groups_;
  }

  specfem::plugin::Points points() const {
    return specfem::plugin::Points(points_);
  }

  Vector position(const int p) const { return { x_[p], y_[p], z_[p] }; }
  Vector normal(const int p) const { return { nx_[p], ny_[p], nz_[p] }; }

  static constexpr int npoints = 5;
  static constexpr int ncomponents = 6;

private:
  std::vector<double> x_, y_, z_, nx_, ny_, nz_, properties_;
  const char *names_[3] = { "kappa", "mu", "rho" };
  specfem_quantity_slot slots_[2] = { { specfem_quantity_velocity, 0, 3 },
                                      { specfem_quantity_traction, 3, 3 } };
  specfem_point_group elastic_{};
  specfem_point_group acoustic_{};
  const specfem_point_group *groups_[2] = { &elastic_, &acoustic_ };
  specfem_points points_{};
};

/// Values [point][component][step] for steps [0, nstep), filled window by
/// window in the given order.
std::vector<double>
fill_windows(const std::vector<std::pair<int, int>> &windows, const int nstep) {
  const Scene scene(granite);
  plane_wave::PlaneWavePlugin plugin(make_config(), make_info());
  plugin.setup(scene.points());

  std::vector<double> values(Scene::npoints * Scene::ncomponents * nstep, 0);
  for (const auto &[first, count] : windows) {
    std::vector<double> window(Scene::npoints * Scene::ncomponents * count);
    specfem_buffer elastic{};
    elastic.struct_size = sizeof(elastic);
    elastic.dtype = specfem_dtype_float64;
    elastic.data = window.data();
    elastic.npoints = Scene::npoints;
    elastic.ncomponents = Scene::ncomponents;
    elastic.nsteps = count;
    elastic.stride_step = 1;
    elastic.stride_component = count;
    elastic.stride_point = Scene::ncomponents * count;
    specfem_buffer acoustic{};
    acoustic.struct_size = sizeof(acoustic);
    acoustic.dtype = specfem_dtype_float64;

    plugin.fill(first, { specfem::plugin::Buffer(elastic),
                         specfem::plugin::Buffer(acoustic) });
    for (int p = 0; p < Scene::npoints; ++p)
      for (int c = 0; c < Scene::ncomponents; ++c)
        for (int s = 0; s < count; ++s)
          values[(p * Scene::ncomponents + c) * nstep + first + s] =
              window[p * elastic.stride_point + c * elastic.stride_component +
                     s];
  }
  return values;
}

void test_plugin_supports() {
  const plane_wave::PlaneWavePlugin plugin(make_config(), make_info());
  using specfem::plugin::Medium;
  using specfem::plugin::Property;
  using specfem::plugin::Quantity;
  CHECK(
      plugin.supports(Medium::elastic, Property::isotropic, Quantity::velocity),
      "elastic isotropic velocity must be supported");
  CHECK(
      plugin.supports(Medium::elastic, Property::isotropic, Quantity::traction),
      "elastic isotropic traction must be supported");
  CHECK(!plugin.supports(Medium::elastic, Property::anisotropic,
                         Quantity::velocity),
        "anisotropic points must be rejected");
  CHECK(!plugin.supports(Medium::acoustic, Property::isotropic,
                         Quantity::potential_rate),
        "acoustic points must be rejected");
}

void test_plugin_rejects_2d() {
  bool thrown = false;
  try {
    const plane_wave::PlaneWavePlugin plugin(make_config(), make_info(2));
  } catch (const specfem::plugin::Unsupported &) {
    thrown = true;
  }
  CHECK(thrown, "a 2D run must be rejected");
}

void test_plugin_material_mismatch() {
  plane_wave::Material other = granite;
  other.vs *= 1.01;
  const Scene scene(other);
  plane_wave::PlaneWavePlugin plugin(make_config(), make_info());
  std::string message;
  try {
    plugin.setup(scene.points());
  } catch (const specfem::plugin::Unsupported &error) {
    message = error.what();
  }
  CHECK(message.find("point 0") != std::string::npos &&
            message.find("vs") != std::string::npos,
        "a material mismatch must name the point and property, got '" << message
                                                                      << "'");
}

void test_plugin_fill() {
  const int nstep = 10;
  const auto whole = fill_windows({ { 0, 10 } }, nstep);
  CHECK(whole == fill_windows({ { 0, 4 }, { 4, 4 }, { 8, 2 } }, nstep),
        "windows of 4 must give the same values as one window");
  CHECK(whole == fill_windows({ { 8, 2 }, { 4, 4 }, { 0, 4 } }, nstep),
        "windows in reverse order must give the same values");

  const Scene scene(granite);
  const auto wave = make_config().plane_wave();
  const auto info = make_info();
  for (int p = 0; p < Scene::npoints; ++p)
    for (int s = 0; s < nstep; ++s) {
      const double t = info.time(s);
      const Vector v = wave.velocity(scene.position(p), t);
      const Vector tr = wave.traction(scene.position(p), scene.normal(p), t);
      for (int c = 0; c < 3; ++c) {
        CHECK(std::abs(whole[(p * Scene::ncomponents + c) * nstep + s] -
                       v[c]) <= 1e-12 * norm(v),
              "slot 0 must hold the velocity");
        CHECK(std::abs(whole[(p * Scene::ncomponents + 3 + c) * nstep + s] -
                       tr[c]) <= 1e-12 * norm(tr),
              "slot 1 must hold the traction");
      }
    }
}

void expect_config_error(const std::string &yaml, const std::string &key) {
  std::string message;
  try {
    plane_wave::parse_config(yaml);
  } catch (const std::runtime_error &error) {
    message = error.what();
  }
  CHECK(message.find(key) != std::string::npos,
        "error must name '" << key << "', got '" << message << "'");
}

void test_config_errors() {
  const std::string material = "material: {rho: 2700, vp: 6000, vs: 3464}\n";
  const std::string stf = "source-time-function: {type: ricker, f0: 2}\n"
                          "material-tolerance: 1.0e-3\n";
  expect_config_error(material +
                          "wave: {type: SH, take-off: 0, azimuth: 0, origin: "
                          "[0, 0, 0], origin-time: 0, amplitude: 1}\n" +
                          stf,
                      "wave.type");
  expect_config_error(material +
                          "wave: {type: P, take-off: 0, origin: [0, 0, 0], "
                          "origin-time: 0, amplitude: 1}\n" +
                          stf,
                      "wave.azimuth");
  expect_config_error(
      material +
          "wave: {type: P, take-off: 0, azimuth: 0, origin: "
          "[0, 0, 0], origin-time: 0, amplitude: 1, speed: 1}\n" +
          stf,
      "wave.speed");
}

} // namespace plane_wave_test

int main() {
  namespace test = plane_wave_test;
  const auto ricker = plane_wave::SourceTimeFunction::ricker(2);
  const auto gaussian = plane_wave::SourceTimeFunction::gaussian(0.1);

  test::test_directions();
  for (const auto type : { plane_wave::WaveType::p, plane_wave::WaveType::sv })
    for (const auto &[stf, stf_name] :
         { std::pair{ ricker, "ricker" }, std::pair{ gaussian, "gaussian" } }) {
      const std::string name =
          std::string(type == plane_wave::WaveType::p ? "P" : "SV") + "/" +
          stf_name;
      test::test_time_derivatives(test::make_wave(type, stf), name);
      test::test_traction(test::make_wave(type, stf), name);
    }
  test::test_sv_is_divergence_free();
  test::test_plugin_supports();
  test::test_plugin_rejects_2d();
  test::test_plugin_material_mismatch();
  test::test_plugin_fill();
  test::test_config_errors();

  if (test::failures > 0) {
    std::cerr << test::failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "All plane-wave checks passed\n";
  return 0;
}
