#include "specfem/injection/fk/model_file_reader.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

std::tuple<specfem::injection::fk::LayeredModel,
           specfem::injection::fk::IncidentWave,
           specfem::injection::fk::TimeWindow>
specfem::injection::fk::read_fk_model_file(const std::string &path) {
  static constexpr double deg_to_rad = 3.141592653589793 / 180.0;
  static constexpr double threshold_vs = 1.0e-6;

  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error(
        "specfem::injection::fk::read_fk_model_file: cannot open file: " +
        path);
  }

  // -------------------------------------------------------------------------
  // State accumulated while reading
  // -------------------------------------------------------------------------
  int nlayer = 0;
  double phi_deg = 0.0;   // azimuth angle in degrees (stored, converted at end)
  double theta_deg = 0.0; // take-off angle in degrees
  double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;
  double origin_time = 0.0;
  double amplitude = 1.0;
  int nstep = 0;
  double deltat = 0.0;
  double frequency_max = 0.1;
  double frequency_sampling = 10.0;
  double time_window = 128.0;
  specfem::injection::fk::incident_wave_type wave_type =
      specfem::injection::fk::incident_wave_type::p;

  // Per-layer data (populated by LAYER lines); sized once NLAYER is known.
  std::vector<double> rho_in, vp_in, vs_in, ztop_in;
  std::vector<bool> layer_set;

  // -------------------------------------------------------------------------
  // Parse loop
  // -------------------------------------------------------------------------
  std::string line;
  while (std::getline(in, line)) {
    // skip blank lines and comment lines
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);
    std::string keyword;
    iss >> keyword;
    if (iss.fail() || keyword.empty())
      continue;

    if (keyword == "NLAYER") {
      iss >> nlayer;
      if (iss.fail() || nlayer <= 0)
        throw std::runtime_error("read_fk_model_file: invalid NLAYER value");
      rho_in.assign(nlayer, 0.0);
      vp_in.assign(nlayer, 0.0);
      vs_in.assign(nlayer, 0.0);
      ztop_in.assign(nlayer + 1,
                     0.0); // ztop_in[nlayer] = dummy (set from last layer)
      layer_set.assign(nlayer, false);

    } else if (keyword == "LAYER") {
      if (nlayer == 0)
        throw std::runtime_error(
            "read_fk_model_file: LAYER keyword before NLAYER");
      int ilayer = 0;
      double rho_l, vp_l, vs_l, ztop_l;
      iss >> ilayer >> rho_l >> vp_l >> vs_l >> ztop_l;
      if (iss.fail() || ilayer < 1 || ilayer > nlayer)
        throw std::runtime_error("read_fk_model_file: malformed LAYER line");
      rho_in[ilayer - 1] = rho_l;
      vp_in[ilayer - 1] = vp_l;
      vs_in[ilayer - 1] = vs_l;
      ztop_in[ilayer - 1] = ztop_l;
      layer_set[ilayer - 1] = true;

    } else if (keyword == "INCIDENT_WAVE") {
      std::string wtype;
      iss >> wtype;
      if (wtype == "p" || wtype == "P") {
        wave_type = specfem::injection::fk::incident_wave_type::p;
      } else if (wtype == "sv" || wtype == "SV") {
        wave_type = specfem::injection::fk::incident_wave_type::sv;
      } else {
        wave_type = specfem::injection::fk::incident_wave_type::p;
      }

    } else if (keyword == "BACK_AZIMUTH") {
      double baz = 0.0;
      iss >> baz;
      phi_deg = -baz - 90.0;

    } else if (keyword == "AZIMUTH") {
      double az = 0.0;
      iss >> az;
      phi_deg = 90.0 - az;

    } else if (keyword == "TAKE_OFF") {
      iss >> theta_deg;

    } else if (keyword == "ORIGIN_WAVEFRONT") {
      iss >> origin_x >> origin_y >> origin_z;
      if (iss.fail())
        throw std::runtime_error(
            "read_fk_model_file: malformed ORIGIN_WAVEFRONT line");

    } else if (keyword == "ORIGIN_TIME") {
      iss >> origin_time;

    } else if (keyword == "NSTEP") {
      iss >> nstep;

    } else if (keyword == "deltat") {
      iss >> deltat;

    } else if (keyword == "FREQUENCY_MAX") {
      iss >> frequency_max;

    } else if (keyword == "FREQUENCY_SAMPLING") {
      iss >> frequency_sampling;

    } else if (keyword == "TIME_WINDOW") {
      iss >> time_window;

    } else if (keyword == "AMPLITUDE") {
      iss >> amplitude;
    }
    // Unknown keywords are silently ignored.
  }

  // -------------------------------------------------------------------------
  // Validate that all layers were supplied
  // -------------------------------------------------------------------------
  if (nlayer == 0)
    throw std::runtime_error("read_fk_model_file: NLAYER not found or is zero");

  for (int i = 0; i < nlayer; ++i) {
    if (!layer_set[i])
      throw std::runtime_error("read_fk_model_file: missing LAYER " +
                               std::to_string(i + 1));
    if (vp_in[i] <= 0.0 || rho_in[i] <= 0.0)
      throw std::runtime_error("read_fk_model_file: layer " +
                               std::to_string(i + 1) +
                               " has non-positive vp or rho");
  }

  // -------------------------------------------------------------------------
  // Compute thicknesses (ref: h_FK[i] = ztop[i] - ztop[i+1])
  // The last layer is the half-space; its ztop[nlayer] mirrors ztop[nlayer-1].
  // -------------------------------------------------------------------------
  ztop_in[nlayer] = ztop_in[nlayer - 1]; // half-space bottom = same as top
  std::vector<double> H(nlayer);
  for (int i = 0; i < nlayer; ++i)
    H[i] = ztop_in[i] - ztop_in[i + 1];

  // -------------------------------------------------------------------------
  // Classify layers: contiguous fluid block on top, elastic block below.
  // A layer with vs < threshold_vs is acoustic.
  // -------------------------------------------------------------------------
  std::vector<specfem::injection::fk::AcousticLayer> fluid_layers;
  std::vector<specfem::injection::fk::ElasticIsotropicLayer> elastic_layers;

  // Determine the split index: all layers 0..nfluid-1 must be fluid,
  // all layers nfluid..nlayer-1 must be elastic.
  int nfluid = 0;
  for (int i = 0; i < nlayer; ++i) {
    if (vs_in[i] < threshold_vs) {
      if (!elastic_layers.empty())
        throw std::runtime_error(
            "read_fk_model_file: fluid layer found below an elastic layer; "
            "fluid layers must form a contiguous top block");
      specfem::injection::fk::AcousticLayer fl;
      fl.density = static_cast<type_real>(rho_in[i]);
      fl.p_velocity = static_cast<type_real>(vp_in[i]);
      fl.thickness = static_cast<type_real>(H[i]);
      fluid_layers.push_back(fl);
      ++nfluid;
    } else {
      specfem::injection::fk::ElasticIsotropicLayer el;
      el.density = static_cast<type_real>(rho_in[i]);
      el.p_velocity = static_cast<type_real>(vp_in[i]);
      el.s_velocity = static_cast<type_real>(vs_in[i]);
      el.thickness = static_cast<type_real>(H[i]);
      elastic_layers.push_back(el);
    }
  }

  if (elastic_layers.empty())
    throw std::runtime_error(
        "read_fk_model_file: no elastic (half-space) layers found");

  // -------------------------------------------------------------------------
  // Build output objects
  // -------------------------------------------------------------------------
  specfem::injection::fk::LayeredModel model(std::move(fluid_layers),
                                             std::move(elastic_layers));

  specfem::injection::fk::IncidentWave wave;
  wave.type = wave_type;
  wave.azimuth_phi = static_cast<type_real>(phi_deg * deg_to_rad);
  wave.take_off_theta = static_cast<type_real>(theta_deg * deg_to_rad);
  wave.origin_x = static_cast<type_real>(origin_x);
  wave.origin_y = static_cast<type_real>(origin_y);
  wave.origin_z = static_cast<type_real>(origin_z);
  wave.origin_time = static_cast<type_real>(origin_time);
  wave.amplitude = static_cast<type_real>(amplitude);
  wave.gaussian_half_duration = 0; // not in file format; default to delta

  specfem::injection::fk::TimeWindow window;
  window.dt = static_cast<type_real>(deltat);
  window.nstep = nstep;
  window.frequency_max = static_cast<type_real>(frequency_max);
  window.frequency_sampling = static_cast<type_real>(frequency_sampling);
  window.time_window_length = static_cast<type_real>(time_window);

  return { model, wave, window };
}
