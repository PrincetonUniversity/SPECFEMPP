#include "plane_wave_response.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace plane_wave_response_tests_impl {

using fk_injection::type_real;

fk_injection::LayeredModel halfspace() {
  fk_injection::ElasticIsotropicLayer layer;
  layer.density = 2700;
  layer.p_velocity = 3000;
  layer.s_velocity = static_cast<type_real>(1732.051);
  return fk_injection::LayeredModel({}, { layer });
}

fk_injection::IncidentWave wave() {
  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.take_off_theta = static_cast<type_real>(20.0 * std::numbers::pi / 180.0);
  wave.azimuth_phi = static_cast<type_real>(0.3);
  wave.origin_z = -20000;
  wave.gaussian_half_duration = 2;
  return wave;
}

fk_injection::TimeWindow window(const double length) {
  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 400;
  window.frequency_sampling = 10;
  window.time_window_length = static_cast<type_real>(length);
  return window;
}

} // namespace plane_wave_response_tests_impl

// The pad keeps the latest corner's abscissae non-negative, so it follows the
// largest delay over the box.
TEST(PlaneWaveResponse, PadFollowsTheLargestCornerDelay) {
  const fk_injection::HorizontalBox box{ -20000, 30000, -5000, 15000 };
  const fk_injection::PlaneWaveResponse response(
      plane_wave_response_tests_impl::halfspace(),
      plane_wave_response_tests_impl::wave(),
      plane_wave_response_tests_impl::window(60), box);

  double time_delay_max = response.time_delay(box.x_min, box.y_min);
  for (const double x : { box.x_min, box.x_max })
    for (const double y : { box.y_min, box.y_max })
      time_delay_max = std::max(time_delay_max, response.time_delay(x, y));
  EXPECT_EQ(
      response.setup().pad_samples,
      static_cast<int>(std::ceil(time_delay_max / response.setup().dt_min)));
}

TEST(PlaneWaveResponse, DelayIsThePlaneWaveDelay) {
  const auto model = plane_wave_response_tests_impl::halfspace();
  const auto wave = plane_wave_response_tests_impl::wave();
  const fk_injection::PlaneWaveResponse response(
      model, wave, plane_wave_response_tests_impl::window(60), {});

  const double p = wave.ray_parameter(model);
  const double eta = std::sqrt(1.0 / (3000.0 * 3000.0) - p * p);
  const double phi = static_cast<double>(wave.azimuth_phi);
  const double x = 7000.0;
  const double y = -3000.0;
  EXPECT_NEAR(response.time_delay(x, y),
              p * (x * std::cos(phi) + y * std::sin(phi)) +
                  eta * -static_cast<double>(wave.origin_z),
              1.0e-9);
}

// The SPECFEM3D convention stretches the delay by N/(N-1).
TEST(PlaneWaveResponse, AbscissaOffsetIsPadMinusStretchedDelay) {
  for (const auto convention :
       { fk_injection::ReferenceConvention::exact,
         fk_injection::ReferenceConvention::specfem3d }) {
    const fk_injection::PlaneWaveResponse response(
        plane_wave_response_tests_impl::halfspace(),
        plane_wave_response_tests_impl::wave(),
        plane_wave_response_tests_impl::window(60), { 0, 10000, 0, 0 },
        convention);
    const auto &setup = response.setup();
    const double fft_samples =
        static_cast<double>(setup.sizes.number_of_fft_samples);
    const double stretch =
        convention == fk_injection::ReferenceConvention::specfem3d
            ? fft_samples / (fft_samples - 1.0)
            : 1.0;
    EXPECT_NEAR(response.abscissa_offset(10000, 0),
                setup.abscissa_pad -
                    response.time_delay(10000, 0) * stretch / setup.dt_min,
                1.0e-9);
  }
}

// A wider box spans more delay, so the window grows to cover it.
TEST(PlaneWaveResponse, WiderBoxLengthensTheWindow) {
  const auto size_for = [](const double half_width) {
    return fk_injection::PlaneWaveResponse(
               plane_wave_response_tests_impl::halfspace(),
               plane_wave_response_tests_impl::wave(),
               plane_wave_response_tests_impl::window(20),
               { -half_width, half_width, -half_width, half_width })
        .setup()
        .sizes.number_of_fft_samples;
  };
  EXPECT_GT(size_for(100000), size_for(0));
}

// A delay span longer than the run outgrows the window's cap.
TEST(PlaneWaveResponse, RejectsABoxTheWindowCannotCover) {
  try {
    const fk_injection::PlaneWaveResponse response(
        plane_wave_response_tests_impl::halfspace(),
        plane_wave_response_tests_impl::wave(),
        plane_wave_response_tests_impl::window(20),
        { -1000000, 1000000, 0, 0 });
    FAIL() << "expected a coverage error";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("sampling abscissa out of range"),
              std::string::npos)
        << error.what();
  }
}

// A take-off past critical in a faster layer makes the wave evanescent there.
TEST(PlaneWaveResponse, RejectsAnEvanescentWave) {
  fk_injection::ElasticIsotropicLayer fast;
  fast.density = 2500;
  fast.p_velocity = 6000;
  fast.s_velocity = 3000;
  fast.thickness = 30000;
  const auto halfspace = plane_wave_response_tests_impl::halfspace();
  auto wave = plane_wave_response_tests_impl::wave();
  wave.take_off_theta =
      static_cast<fk_injection::type_real>(60.0 * std::numbers::pi / 180.0);
  EXPECT_THROW(fk_injection::PlaneWaveResponse(
                   fk_injection::LayeredModel(
                       {}, { fast, halfspace.elastic_layers().back() }),
                   wave, plane_wave_response_tests_impl::window(60), {}),
               specfem::plugin::Unsupported);
}
