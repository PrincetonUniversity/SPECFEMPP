#include "config.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>

namespace plane_wave::config_impl {

std::runtime_error config_error(const std::string &key,
                                const std::string &message) {
  return std::runtime_error("plane-wave config: " + key + ": " + message);
}

YAML::Node require_map(const YAML::Node &parent, const std::string &key,
                       const std::string &path) {
  const YAML::Node node = parent[key];
  if (!node)
    throw config_error(path, "missing");
  if (!node.IsMap())
    throw config_error(path, "must be a map");
  return node;
}

void reject_unknown_keys(const YAML::Node &node, const std::string &path,
                         std::initializer_list<const char *> known) {
  for (const auto &entry : node) {
    const auto key = entry.first.as<std::string>();
    bool found = false;
    for (const char *name : known)
      found = found || key == name;
    if (!found)
      throw config_error(path.empty() ? key : path + "." + key, "unknown key");
  }
}

double require_number(const YAML::Node &parent, const std::string &key,
                      const std::string &path) {
  const YAML::Node node = parent[key];
  if (!node)
    throw config_error(path, "missing");
  try {
    return node.as<double>();
  } catch (const YAML::Exception &) {
    throw config_error(path, "must be a number");
  }
}

double require_positive(const YAML::Node &parent, const std::string &key,
                        const std::string &path) {
  const double value = require_number(parent, key, path);
  if (!(value > 0))
    throw config_error(path, "must be positive");
  return value;
}

} // namespace plane_wave::config_impl

plane_wave::Config plane_wave::parse_config(const std::string &text) {
  namespace impl = plane_wave::config_impl;
  YAML::Node root;
  try {
    root = YAML::Load(text);
  } catch (const YAML::Exception &error) {
    throw std::runtime_error("plane-wave config: invalid YAML: " +
                             std::string(error.what()));
  }
  if (!root.IsMap())
    throw std::runtime_error("plane-wave config: expected a map");
  impl::reject_unknown_keys(
      root, "",
      { "material", "wave", "source-time-function", "material-tolerance" });

  plane_wave::Config config;

  const YAML::Node material = impl::require_map(root, "material", "material");
  impl::reject_unknown_keys(material, "material", { "rho", "vp", "vs" });
  config.material.rho = impl::require_positive(material, "rho", "material.rho");
  config.material.vp = impl::require_positive(material, "vp", "material.vp");
  config.material.vs = impl::require_positive(material, "vs", "material.vs");
  if (!(config.material.vs < config.material.vp))
    throw impl::config_error("material.vs", "must be smaller than material.vp");

  const YAML::Node wave = impl::require_map(root, "wave", "wave");
  impl::reject_unknown_keys(
      wave, "wave",
      { "type", "take-off", "azimuth", "origin", "origin-time", "amplitude" });
  if (!wave["type"])
    throw impl::config_error("wave.type", "missing");
  const auto type = wave["type"].as<std::string>();
  if (type == "P")
    config.type = plane_wave::WaveType::p;
  else if (type == "SV")
    config.type = plane_wave::WaveType::sv;
  else
    throw impl::config_error("wave.type",
                             "must be P or SV, got '" + type + "'");
  config.take_off = impl::require_number(wave, "take-off", "wave.take-off");
  config.azimuth = impl::require_number(wave, "azimuth", "wave.azimuth");
  const YAML::Node origin = wave["origin"];
  if (!origin)
    throw impl::config_error("wave.origin", "missing");
  if (!origin.IsSequence() || origin.size() != 3)
    throw impl::config_error("wave.origin", "must be a list of 3 numbers");
  for (int i = 0; i < 3; ++i)
    config.origin[i] = origin[i].as<double>();
  config.origin_time =
      impl::require_number(wave, "origin-time", "wave.origin-time");
  config.amplitude = impl::require_number(wave, "amplitude", "wave.amplitude");

  const YAML::Node stf =
      impl::require_map(root, "source-time-function", "source-time-function");
  if (!stf["type"])
    throw impl::config_error("source-time-function.type", "missing");
  const auto kind = stf["type"].as<std::string>();
  if (kind == "ricker") {
    impl::reject_unknown_keys(stf, "source-time-function", { "type", "f0" });
    config.waveform = plane_wave::SourceTimeFunction::ricker(
        impl::require_positive(stf, "f0", "source-time-function.f0"));
  } else if (kind == "gaussian") {
    impl::reject_unknown_keys(stf, "source-time-function",
                              { "type", "half-duration" });
    config.waveform =
        plane_wave::SourceTimeFunction::gaussian(impl::require_positive(
            stf, "half-duration", "source-time-function.half-duration"));
  } else {
    throw impl::config_error("source-time-function.type",
                             "must be ricker or gaussian, got '" + kind + "'");
  }

  config.material_tolerance =
      impl::require_positive(root, "material-tolerance", "material-tolerance");
  return config;
}

plane_wave::Config plane_wave::load_config(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("plane-wave config: cannot open '" + path + "'");
  std::stringstream text;
  text << file.rdbuf();
  return plane_wave::parse_config(text.str());
}
