#include "plugin_config.hpp"
#include "fk_injection/model_file_reader.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fk_injection::plugin_config_impl {

std::string key_path(const std::string &prefix, const std::string &key);
std::runtime_error config_error(const std::string &key,
                                const std::string &message);
void reject_unknown_keys(const YAML::Node &node, const std::string &prefix,
                         std::initializer_list<const char *> known);
YAML::Node require_map(const YAML::Node &parent, const std::string &key,
                       const std::string &prefix);
std::optional<double> optional_number(const YAML::Node &parent,
                                      const std::string &key,
                                      const std::string &prefix);
double require_number(const YAML::Node &parent, const std::string &key,
                      const std::string &prefix);
fk_injection::LayeredModel parse_layers(const YAML::Node &root);
fk_injection::PluginConfig read_model_file(const YAML::Node &root);
void parse_incidence(const YAML::Node &incidence, bool all_required,
                     fk_injection::PluginConfig &config);
void parse_time_window(const YAML::Node &window, bool all_required,
                       fk_injection::PluginConfig &config);

} // namespace fk_injection::plugin_config_impl

std::string
fk_injection::plugin_config_impl::key_path(const std::string &prefix,
                                           const std::string &key) {
  return prefix.empty() ? key : prefix + "." + key;
}

std::runtime_error
fk_injection::plugin_config_impl::config_error(const std::string &key,
                                               const std::string &message) {
  return std::runtime_error("fk config: " + key + ": " + message);
}

void fk_injection::plugin_config_impl::reject_unknown_keys(
    const YAML::Node &node, const std::string &prefix,
    std::initializer_list<const char *> known) {
  for (const auto &entry : node) {
    const auto key = entry.first.as<std::string>();
    bool found = false;
    for (const char *name : known)
      found = found || key == name;
    if (!found)
      throw config_error(key_path(prefix, key), "unknown key");
  }
}

YAML::Node
fk_injection::plugin_config_impl::require_map(const YAML::Node &parent,
                                              const std::string &key,
                                              const std::string &prefix) {
  const YAML::Node node = parent[key];
  if (!node)
    throw config_error(key_path(prefix, key), "missing");
  if (!node.IsMap())
    throw config_error(key_path(prefix, key), "must be a map");
  return node;
}

std::optional<double>
fk_injection::plugin_config_impl::optional_number(const YAML::Node &parent,
                                                  const std::string &key,
                                                  const std::string &prefix) {
  const YAML::Node node = parent[key];
  if (!node)
    return std::nullopt;
  try {
    return node.as<double>();
  } catch (const YAML::Exception &) {
    throw config_error(key_path(prefix, key), "must be a number");
  }
}

double
fk_injection::plugin_config_impl::require_number(const YAML::Node &parent,
                                                 const std::string &key,
                                                 const std::string &prefix) {
  const std::optional<double> value = optional_number(parent, key, prefix);
  if (!value)
    throw config_error(key_path(prefix, key), "missing");
  return *value;
}

fk_injection::LayeredModel
fk_injection::plugin_config_impl::parse_layers(const YAML::Node &root) {
  const YAML::Node layers = root["layers"];
  if (!layers.IsSequence())
    throw config_error("layers", "must be a list");

  std::vector<fk_injection::AcousticLayer> fluid_layers;
  std::vector<fk_injection::ElasticIsotropicLayer> elastic_layers;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const std::string path = "layers[" + std::to_string(i) + "]";
    const YAML::Node entry = layers[i];
    if (!entry.IsMap())
      throw config_error(path, "must be a map");
    reject_unknown_keys(entry, path, { "acoustic", "elastic" });
    if (entry.size() != 1)
      throw config_error(path, "must be one of acoustic or elastic");

    const bool is_acoustic = static_cast<bool>(entry["acoustic"]);
    if (is_acoustic && !elastic_layers.empty())
      throw config_error(path, "acoustic layers must form a contiguous "
                               "block above the elastic layers");
    const std::string kind = is_acoustic ? "acoustic" : "elastic";
    const std::string layer_path = key_path(path, kind);
    const YAML::Node layer = require_map(entry, kind, path);
    const auto number = [&](const char *key) {
      return static_cast<type_real>(require_number(layer, key, layer_path));
    };

    if (is_acoustic) {
      reject_unknown_keys(layer, layer_path, { "rho", "vp", "thickness" });
      fluid_layers.push_back({ .density = number("rho"),
                               .p_velocity = number("vp"),
                               .thickness = number("thickness") });
    } else {
      reject_unknown_keys(layer, layer_path,
                          { "rho", "vp", "vs", "thickness" });
      elastic_layers.push_back({ .density = number("rho"),
                                 .p_velocity = number("vp"),
                                 .s_velocity = number("vs"),
                                 .thickness = number("thickness") });
    }
  }

  try {
    return fk_injection::LayeredModel(std::move(fluid_layers),
                                      std::move(elastic_layers));
  } catch (const std::runtime_error &error) {
    throw config_error("layers", error.what());
  }
}

fk_injection::PluginConfig
fk_injection::plugin_config_impl::read_model_file(const YAML::Node &root) {
  try {
    auto [model, wave, window, has_origin] =
        fk_injection::read_fk_model_file(root["model-file"].as<std::string>());
    if (wave.origin_time != 0)
      throw std::runtime_error("a non-zero ORIGIN_TIME is not supported; set "
                               "incidence.origin-time, the simulation time the "
                               "wavefront crosses the origin");
    const YAML::Node incidence = root["incidence"];
    if (!has_origin && !(incidence && incidence.IsMap() && incidence["origin"]))
      throw std::runtime_error("no ORIGIN_WAVEFRONT, and the plugin does not "
                               "derive one as SPECFEM3D does; add it or set "
                               "incidence.origin");
    return fk_injection::PluginConfig{ .model = std::move(model),
                                       .mesh_frame_wave = wave,
                                       .frequency_max = window.frequency_max,
                                       .frequency_sampling =
                                           window.frequency_sampling,
                                       .time_window_length =
                                           window.time_window_length };
  } catch (const std::runtime_error &error) {
    throw config_error("model-file", error.what());
  }
}

void fk_injection::plugin_config_impl::parse_incidence(
    const YAML::Node &incidence, const bool all_required,
    fk_injection::PluginConfig &config) {
  static constexpr double deg_to_rad = 3.141592653589793 / 180.0;
  auto &wave = config.mesh_frame_wave;

  reject_unknown_keys(incidence, "incidence",
                      { "type", "back-azimuth", "azimuth", "take-off", "origin",
                        "origin-time", "amplitude", "half-duration" });

  if (incidence["type"]) {
    auto type = incidence["type"].as<std::string>();
    std::transform(type.begin(), type.end(), type.begin(),
                   [](const unsigned char c) { return std::tolower(c); });
    if (type == "p")
      wave.type = fk_injection::incident_wave_type::p;
    else if (type == "sv")
      wave.type = fk_injection::incident_wave_type::sv;
    else
      throw config_error("incidence.type",
                         "must be P or SV, got '" + type + "'");
  } else if (all_required) {
    throw config_error("incidence.type", "missing");
  }

  const auto back_azimuth =
      optional_number(incidence, "back-azimuth", "incidence");
  const auto azimuth = optional_number(incidence, "azimuth", "incidence");
  if ((back_azimuth && azimuth) || (all_required && !back_azimuth && !azimuth))
    throw config_error("incidence",
                       "exactly one of back-azimuth or azimuth must be given");
  if (back_azimuth)
    wave.azimuth_phi =
        static_cast<type_real>((-*back_azimuth - 90.0) * deg_to_rad);
  else if (azimuth)
    wave.azimuth_phi = static_cast<type_real>((90.0 - *azimuth) * deg_to_rad);

  if (const auto take_off = optional_number(incidence, "take-off", "incidence"))
    wave.take_off_theta = static_cast<type_real>(*take_off * deg_to_rad);
  else if (all_required)
    throw config_error("incidence.take-off", "missing");

  if (const YAML::Node origin = incidence["origin"]) {
    try {
      const auto xyz = origin.as<std::array<double, 3>>();
      wave.origin_x = static_cast<type_real>(xyz[0]);
      wave.origin_y = static_cast<type_real>(xyz[1]);
      wave.origin_z = static_cast<type_real>(xyz[2]);
    } catch (const YAML::Exception &) {
      throw config_error("incidence.origin", "must be a list of 3 numbers");
    }
  } else if (all_required) {
    throw config_error("incidence.origin", "missing");
  }

  config.origin_time = optional_number(incidence, "origin-time", "incidence");

  if (const auto amplitude =
          optional_number(incidence, "amplitude", "incidence"))
    wave.amplitude = static_cast<type_real>(*amplitude);

  if (const auto half_duration =
          optional_number(incidence, "half-duration", "incidence")) {
    if (*half_duration < 0)
      throw config_error("incidence.half-duration", "must not be negative");
    wave.gaussian_half_duration = static_cast<type_real>(*half_duration);
  }
}

void fk_injection::plugin_config_impl::parse_time_window(
    const YAML::Node &window, const bool all_required,
    fk_injection::PluginConfig &config) {
  reject_unknown_keys(window, "time-window",
                      { "frequency-max", "frequency-sampling", "length" });
  if (const auto value =
          optional_number(window, "frequency-max", "time-window"))
    config.frequency_max = static_cast<type_real>(*value);
  if (const auto value =
          optional_number(window, "frequency-sampling", "time-window"))
    config.frequency_sampling = static_cast<type_real>(*value);
  if (const auto value = optional_number(window, "length", "time-window"))
    config.time_window_length = static_cast<type_real>(*value);
  else if (all_required)
    throw config_error("time-window.length", "missing");
}

fk_injection::TimeWindow
fk_injection::PluginConfig::time_window(const type_real dt,
                                        const int nstep) const {
  fk_injection::TimeWindow window;
  window.dt = dt;
  window.nstep = nstep;
  window.frequency_max = frequency_max;
  window.frequency_sampling = frequency_sampling;
  window.time_window_length = time_window_length;
  return window;
}

fk_injection::IncidentWave
fk_injection::PluginConfig::incident_wave(const double t0,
                                          const double zmax) const {
  fk_injection::IncidentWave result = mesh_frame_wave;
  result.origin_time = static_cast<type_real>(t0 - origin_time.value_or(t0));
  result.origin_z = static_cast<type_real>(
      static_cast<double>(mesh_frame_wave.origin_z) - zmax);
  return result;
}

fk_injection::PluginConfig
fk_injection::parse_plugin_config(const std::string &text) {
  namespace impl = fk_injection::plugin_config_impl;

  YAML::Node root;
  try {
    root = YAML::Load(text);
  } catch (const YAML::Exception &error) {
    throw std::runtime_error("fk config: invalid YAML: " +
                             std::string(error.what()));
  }
  if (!root.IsMap())
    throw std::runtime_error("fk config: expected a map");
  impl::reject_unknown_keys(root, "",
                            { "layers", "model-file", "incidence",
                              "time-window", "material-tolerance",
                              "reproduce-specfem3d-reference" });

  const bool has_layers = static_cast<bool>(root["layers"]);
  if (has_layers == static_cast<bool>(root["model-file"]))
    throw std::runtime_error(
        "fk config: exactly one of layers or model-file must be given");

  fk_injection::PluginConfig config =
      has_layers
          ? fk_injection::PluginConfig{ .model = impl::parse_layers(root) }
          : impl::read_model_file(root);
  auto &wave = config.mesh_frame_wave;

  bool half_duration_set = false;
  if (has_layers || root["incidence"]) {
    const YAML::Node incidence = impl::require_map(root, "incidence", "");
    impl::parse_incidence(incidence, has_layers, config);
    half_duration_set = static_cast<bool>(incidence["half-duration"]);
  }

  if (has_layers || root["time-window"])
    impl::parse_time_window(impl::require_map(root, "time-window", ""),
                            has_layers, config);

  // The half-duration Tg and frequency-max are linked by Tg = 1/frequency-max,
  // as in the model file reader. Precedence: YAML > model file > derived.
  if (!half_duration_set && wave.gaussian_half_duration <= 0 &&
      config.frequency_max > 0)
    wave.gaussian_half_duration =
        static_cast<type_real>(1.0 / static_cast<double>(config.frequency_max));
  if (config.frequency_max <= 0 && wave.gaussian_half_duration > 0)
    config.frequency_max = static_cast<type_real>(
        1.0 / static_cast<double>(wave.gaussian_half_duration));
  if (config.frequency_max <= 0 && wave.gaussian_half_duration <= 0)
    throw std::runtime_error("fk config: no band limit: give "
                             "time-window.frequency-max or "
                             "incidence.half-duration");
  // Storage sampling frequency defaults to 10/Tg (dt_min ~ Tg/10).
  if (config.frequency_sampling <= 0 && wave.gaussian_half_duration > 0)
    config.frequency_sampling = static_cast<type_real>(
        10.0 / static_cast<double>(wave.gaussian_half_duration));

  // The spectrum exp(-(omega Tg / 2)^2) has energy density exp(-(omega
  // Tg)^2/2).
  static constexpr double maximum_energy_above_band = 0.4;
  const double energy_above_band =
      std::erfc(std::sqrt(2.0) * 3.141592653589793 *
                static_cast<double>(config.frequency_max) *
                static_cast<double>(wave.gaussian_half_duration));
  if (energy_above_band > maximum_energy_above_band)
    throw impl::config_error(
        "incidence.half-duration",
        std::to_string(100.0 * energy_above_band) +
            "% of the Gaussian's energy is above time-window.frequency-max; "
            "give a longer half-duration or a higher frequency-max");

  if (const auto value =
          impl::optional_number(root, "material-tolerance", "")) {
    if (!(*value > 0))
      throw impl::config_error("material-tolerance", "must be positive");
    config.material_tolerance = *value;
  }

  if (const YAML::Node reproduce = root["reproduce-specfem3d-reference"]) {
    try {
      if (reproduce.as<bool>())
        config.convention = fk_injection::ReferenceConvention::specfem3d;
    } catch (const YAML::Exception &) {
      throw impl::config_error("reproduce-specfem3d-reference",
                               "must be true or false");
    }
  }

  return config;
}

fk_injection::PluginConfig
fk_injection::load_plugin_config(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("fk config: cannot open '" + path + "'");
  std::stringstream text;
  text << file.rdbuf();
  return fk_injection::parse_plugin_config(text.str());
}
