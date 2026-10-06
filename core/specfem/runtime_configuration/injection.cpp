#include "injection.hpp"
#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/model_file_reader.hpp"
#include "specfem/injection/fk_provider.hpp"
#include "specfem/injection/injection_provider.hpp"
#include "specfem/utilities/strings.hpp"
#include <stdexcept>
#include <tuple>
#include <vector>

specfem::runtime_configuration::Injection::Injection(const YAML::Node &node) {
  // Parse `enabled` (optional, default true).
  if (node["enabled"]) {
    try {
      enabled_ = node["enabled"].as<bool>();
    } catch (const YAML::Exception &e) {
      throw std::runtime_error(
          std::string("Error parsing injection configuration: ") + e.what());
    }
  }

  // Parse `method` (optional, default "fk").
  if (node["method"]) {
    try {
      method_ = node["method"].as<std::string>();
    } catch (const YAML::Exception &e) {
      throw std::runtime_error(
          std::string("Error parsing injection configuration: ") + e.what());
    }
    method_ = specfem::utilities::to_lower(method_);
  }

  // Exactly one of `layers` / `model-file` must be present.
  const bool has_layers = static_cast<bool>(node["layers"]);
  const bool has_model_file = static_cast<bool>(node["model-file"]);

  if (has_layers && has_model_file) {
    throw std::runtime_error(
        "injection: exactly one of 'layers' or 'model-file' must be specified");
  }
  if (!has_layers && !has_model_file) {
    throw std::runtime_error(
        "injection: exactly one of 'layers' or 'model-file' must be specified");
  }

  if (has_layers) {
    // -----------------------------------------------------------------------
    // Inline form: parse the `layers:` sequence.
    // -----------------------------------------------------------------------
    try {
      const YAML::Node &layers_node = node["layers"];
      if (!layers_node.IsSequence()) {
        throw std::runtime_error("injection: 'layers' must be a YAML sequence");
      }

      std::vector<specfem::injection::fk::AcousticLayer> fluid_layers;
      std::vector<specfem::injection::fk::ElasticIsotropicLayer> elastic_layers;

      for (std::size_t i = 0; i < layers_node.size(); ++i) {
        const YAML::Node &entry = layers_node[i];

        if (entry["acoustic"]) {
          const YAML::Node &ac = entry["acoustic"];
          if (ac["vs"]) {
            throw std::runtime_error(
                "injection: acoustic layer must not specify 'vs'");
          }
          specfem::injection::fk::AcousticLayer fl;
          fl.density = ac["rho"].as<type_real>();
          fl.p_velocity = ac["vp"].as<type_real>();
          fl.thickness = ac["thickness"].as<type_real>();
          fluid_layers.push_back(fl);

        } else if (entry["elastic"]) {
          const YAML::Node &el = entry["elastic"];
          if (!el["vs"]) {
            throw std::runtime_error(
                "injection: elastic layer must specify 'vs'");
          }
          specfem::injection::fk::ElasticIsotropicLayer layer;
          layer.density = el["rho"].as<type_real>();
          layer.p_velocity = el["vp"].as<type_real>();
          layer.s_velocity = el["vs"].as<type_real>();
          layer.thickness = el["thickness"].as<type_real>();
          elastic_layers.push_back(layer);

        } else {
          // Report the offending key if we can find one, else report index.
          std::string offending_key = "(unknown)";
          if (entry.IsMap() && entry.begin() != entry.end()) {
            offending_key = entry.begin()->first.as<std::string>();
          }
          throw std::runtime_error("injection: unknown layer type key '" +
                                   offending_key + "'");
        }
      }

      model_ = specfem::injection::fk::LayeredModel(std::move(fluid_layers),
                                                    std::move(elastic_layers));
    } catch (const std::runtime_error &) {
      throw; // propagate validation errors as-is
    } catch (const YAML::Exception &e) {
      throw std::runtime_error(
          std::string("Error parsing injection configuration: ") + e.what());
    }

  } else {
    // -----------------------------------------------------------------------
    // File form: delegate to read_fk_model_file.
    // -----------------------------------------------------------------------
    try {
      const auto path = node["model-file"].as<std::string>();
      auto [m, w, win] = specfem::injection::fk::read_fk_model_file(path);
      model_ = m;
      wave_ = w;
      frequency_max_ = win.frequency_max;
      frequency_sampling_ = win.frequency_sampling;
      time_window_length_ = win.time_window_length;
    } catch (const std::runtime_error &) {
      throw;
    } catch (const YAML::Exception &e) {
      throw std::runtime_error(
          std::string("Error parsing injection configuration: ") + e.what());
    }
  }

  // -----------------------------------------------------------------------
  // `incidence:` block.  Required for the inline form; optional override for
  // the file form.
  // -----------------------------------------------------------------------
  if (node["incidence"]) {
    try {
      static constexpr double deg_to_rad = 3.141592653589793 / 180.0;

      const YAML::Node &inc = node["incidence"];

      // Wave type.
      const std::string type_str =
          specfem::utilities::to_lower(inc["type"].as<std::string>());
      if (type_str == "p") {
        wave_.type = specfem::injection::fk::incident_wave_type::p;
      } else if (type_str == "sv") {
        wave_.type = specfem::injection::fk::incident_wave_type::sv;
      } else {
        throw std::runtime_error(
            "injection: incidence type must be 'P' or 'SV'");
      }

      // Azimuth-phi convention (matches model_file_reader.cpp).
      double phi_deg = 0.0;
      if (inc["back-azimuth"]) {
        const double baz = inc["back-azimuth"].as<double>();
        phi_deg = -baz - 90.0;
      } else if (inc["azimuth"]) {
        const double az = inc["azimuth"].as<double>();
        phi_deg = 90.0 - az;
      }
      wave_.azimuth_phi = static_cast<type_real>(phi_deg * deg_to_rad);

      // Take-off angle (degrees -> radians).
      if (inc["take-off"]) {
        const double theta_deg = inc["take-off"].as<double>();
        wave_.take_off_theta = static_cast<type_real>(theta_deg * deg_to_rad);
      }

      // Origin coordinates.
      if (inc["origin"]) {
        const YAML::Node &orig = inc["origin"];
        wave_.origin_x = static_cast<type_real>(orig[0].as<double>());
        wave_.origin_y = static_cast<type_real>(orig[1].as<double>());
        wave_.origin_z = static_cast<type_real>(orig[2].as<double>());
      }

      // Origin time.
      if (inc["origin-time"]) {
        wave_.origin_time =
            static_cast<type_real>(inc["origin-time"].as<double>());
      }

      // Amplitude.
      if (inc["amplitude"]) {
        wave_.amplitude = static_cast<type_real>(inc["amplitude"].as<double>());
      }

      // Gaussian half-duration.
      if (inc["half-duration"]) {
        wave_.gaussian_half_duration =
            static_cast<type_real>(inc["half-duration"].as<double>());
      }

    } catch (const std::runtime_error &) {
      throw;
    } catch (const YAML::Exception &e) {
      throw std::runtime_error(
          std::string("Error parsing injection configuration: ") + e.what());
    }

  } else if (has_layers) {
    throw std::runtime_error(
        "injection: inline 'layers' form requires an 'incidence' block");
  }

  // -----------------------------------------------------------------------
  // `time-window:` block.  Required for the inline form; optional override
  // for the file form.
  // -----------------------------------------------------------------------
  if (node["time-window"]) {
    try {
      const YAML::Node &tw = node["time-window"];
      frequency_max_ = tw["frequency-max"].as<type_real>();
      frequency_sampling_ = tw["frequency-sampling"].as<type_real>();
      time_window_length_ = tw["length"].as<type_real>();
    } catch (const std::runtime_error &) {
      throw;
    } catch (const YAML::Exception &e) {
      throw std::runtime_error(
          std::string("Error parsing injection configuration: ") + e.what());
    }
  } else if (has_layers) {
    throw std::runtime_error(
        "injection: inline 'layers' form requires a 'time-window' block");
  }
}

specfem::injection::fk::TimeWindow
specfem::runtime_configuration::Injection::to_time_window(type_real dt,
                                                          int nstep) const {
  specfem::injection::fk::TimeWindow window;
  window.dt = dt;
  window.nstep = nstep;
  window.frequency_max = frequency_max_;
  window.frequency_sampling = frequency_sampling_;
  window.time_window_length = time_window_length_;
  return window;
}

template <specfem::element::dimension_tag DimensionTag>
std::shared_ptr<specfem::injection::injection_provider<DimensionTag>>
specfem::runtime_configuration::Injection::instantiate(
    const specfem::injection::fk::EvalPoints &points, type_real dt, int nstep,
    bool compute_traction) const {
  if (method_ == "fk") {
    const auto window = this->to_time_window(dt, nstep);
    return std::make_shared<specfem::injection::fk_provider<DimensionTag>>(
        model_, wave_, window, points, compute_traction);
  }
  throw std::runtime_error("injection: unknown method '" + method_ +
                           "'. Supported methods: fk");
}

// Explicit instantiations for both spatial dimensions.
template std::shared_ptr<specfem::injection::injection_provider<
    specfem::element::dimension_tag::dim2>>
specfem::runtime_configuration::Injection::instantiate<
    specfem::element::dimension_tag::dim2>(
    const specfem::injection::fk::EvalPoints &, type_real, int, bool) const;

template std::shared_ptr<specfem::injection::injection_provider<
    specfem::element::dimension_tag::dim3>>
specfem::runtime_configuration::Injection::instantiate<
    specfem::element::dimension_tag::dim3>(
    const specfem::injection::fk::EvalPoints &, type_real, int, bool) const;
