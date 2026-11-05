#include "fk_injection_plugin.hpp"
#include "plane_wave_response.hpp"
#include "series_table.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

fk_injection::FkInjectionPlugin::FkInjectionPlugin(
    const std::string &config_path, const specfem::plugin::SimulationInfo &info,
    const specfem::plugin::Communicator &)
    : FkInjectionPlugin(fk_injection::load_plugin_config(config_path), info) {}

fk_injection::FkInjectionPlugin::FkInjectionPlugin(
    const fk_injection::PluginConfig &config,
    const specfem::plugin::SimulationInfo &info)
    : config_(config), wave_(config.incident_wave(info.t0(), info.bounds()[5])),
      window_(config.time_window(static_cast<type_real>(info.dt()),
                                 info.number_of_steps())),
      zmax_(info.bounds()[5]), box_{ info.bounds()[0], info.bounds()[1],
                                     info.bounds()[2], info.bounds()[3] },
      depth_tolerance_(fk_injection::SeriesTable::depth_tolerance(
          std::max(std::abs(info.bounds()[4]), std::abs(info.bounds()[5])))) {}

bool fk_injection::FkInjectionPlugin::supports(
    const specfem::plugin::Medium medium,
    const specfem::plugin::Property property,
    const specfem::plugin::Quantity quantity) const {
  if (property != specfem::plugin::Property::isotropic)
    return false;
  switch (medium) {
  case specfem::plugin::Medium::elastic:
    return quantity == specfem::plugin::Quantity::velocity ||
           quantity == specfem::plugin::Quantity::traction;
  case specfem::plugin::Medium::acoustic:
    return quantity == specfem::plugin::Quantity::displacement ||
           quantity == specfem::plugin::Quantity::potential_rate;
  default:
    return false;
  }
}

void fk_injection::FkInjectionPlugin::setup(
    const specfem::plugin::Points &points) {
  groups_.clear();
  std::vector<std::vector<fk_injection::FieldPoint>> group_points;
  std::vector<fk_injection::SeriesKey> keys;
  for (const auto &group : points.groups()) {
    auto &field_points = group_points.emplace_back();
    if (group.slots().empty())
      continue;
    for (std::int64_t p = 0; p < group.size(); ++p) {
      const std::int64_t i = group.begin() + p;
      const double depth = zmax_ - points.z(i);

      fk_injection::FieldPoint point;
      point.x = static_cast<type_real>(points.x(i));
      point.y = static_cast<type_real>(points.y(i));
      point.depth = static_cast<type_real>(depth);
      point.layer = material_layer(points, i, group.medium(), depth,
                                   point_material(group, p));
      point.normal_x = static_cast<type_real>(points.nx(i));
      point.normal_y = static_cast<type_real>(points.ny(i));
      point.normal_z = static_cast<type_real>(points.nz(i));
      field_points.push_back(point);
      keys.push_back({ point.layer, point.depth });
    }
  }

  const fk_injection::PlaneWaveResponse response(config_.model, wave_, window_,
                                                 box_, config_.convention);
  const auto series = std::make_shared<const fk_injection::SeriesTable>(
      fk_injection::SeriesTable::synthesize(response, keys, depth_tolerance_));

  std::size_t largest_group = 0;
  const auto &sdk_groups = points.groups();
  for (std::size_t g = 0; g < group_points.size(); ++g) {
    largest_group = std::max(largest_group, group_points[g].size());
    groups_.push_back(
        { fk_injection::FieldSampler(response, series, group_points[g]),
          sdk_groups[g].slots() });
  }
  scratch_.resize(largest_group * fk_injection::number_of_sample_components);
}

fk_injection::FkInjectionPlugin::Material
fk_injection::FkInjectionPlugin::point_material(
    const specfem::plugin::PointGroup &group, const std::int64_t point) {
  if (group.medium() == specfem::plugin::Medium::acoustic) {
    const double rho_inverse = group.property(point, "rho_inverse");
    return { 1.0 / rho_inverse,
             std::sqrt(group.property(point, "kappa") * rho_inverse), 0.0 };
  }
  const double rho = group.property(point, "rho");
  const double mu = group.property(point, "mu");
  const double kappa = group.property(point, "kappa");
  return { rho, std::sqrt((kappa + 4.0 * mu / 3.0) / rho),
           std::sqrt(mu / rho) };
}

fk_injection::FkInjectionPlugin::Material
fk_injection::FkInjectionPlugin::layer_material(const int layer) const {
  const fk_injection::Layer &model_layer = config_.model.layers()[layer];
  if (const auto *fluid =
          std::get_if<fk_injection::AcousticLayer>(&model_layer))
    return { static_cast<double>(fluid->density),
             static_cast<double>(fluid->p_velocity), 0.0 };
  const auto &solid =
      std::get<fk_injection::ElasticIsotropicLayer>(model_layer);
  return { static_cast<double>(solid.density),
           static_cast<double>(solid.p_velocity),
           static_cast<double>(solid.s_velocity) };
}

int fk_injection::FkInjectionPlugin::material_layer(
    const specfem::plugin::Points &points, const std::int64_t index,
    const specfem::plugin::Medium medium, const double depth,
    const Material &material) const {
  const auto describe_point = [&] {
    std::ostringstream message;
    message << "fk: point " << index << " at (" << points.x(index) << ", "
            << points.y(index) << ", " << points.z(index) << ")";
    return message.str();
  };

  fk_injection::LayerLocation location;
  try {
    location = config_.model.locate(depth, medium, depth_tolerance_);
  } catch (const std::runtime_error &error) {
    throw specfem::plugin::Unsupported(describe_point() + ": " + error.what());
  }

  const double tolerance = config_.material_tolerance;
  const auto close = [tolerance](const double got, const double want) {
    return std::abs(got - want) <= tolerance * std::abs(want);
  };
  const auto matches = [&](const Material &layer) {
    return close(material.rho, layer.rho) && close(material.vp, layer.vp) &&
           close(material.vs, layer.vs);
  };

  const Material layer = layer_material(location.layer);
  if (matches(layer))
    return location.layer;

  // A GLL point on a layer interface carries its own element's material, which
  // can be the lower layer's. The field is continuous there, so the point
  // reads the top of that layer.
  const auto &layers = config_.model.layers();
  if (location.height <= depth_tolerance_ &&
      location.layer + 1 < config_.model.number_of_layers() &&
      layers[location.layer + 1].index() == layers[location.layer].index() &&
      matches(layer_material(location.layer + 1)))
    return location.layer + 1;

  std::ostringstream message;
  message << describe_point() << " has (rho, vp, vs) = (" << material.rho
          << ", " << material.vp << ", " << material.vs
          << "), but the model layer at depth " << depth << " has ("
          << layer.rho << ", " << layer.vp << ", " << layer.vs
          << ") (relative tolerance " << tolerance << ")";
  throw specfem::plugin::Unsupported(message.str());
}

int fk_injection::FkInjectionPlugin::sample_offset(
    const specfem::plugin::Quantity quantity) {
  switch (quantity) {
  case specfem::plugin::Quantity::velocity:
  case specfem::plugin::Quantity::displacement:
    return fk_injection::sample_kinematic_offset;
  case specfem::plugin::Quantity::traction:
    return fk_injection::sample_traction_offset;
  case specfem::plugin::Quantity::potential_rate:
    return fk_injection::sample_potential_rate_offset;
  default:
    throw std::logic_error("fk: unsupported quantity reached fill");
  }
}

void fk_injection::FkInjectionPlugin::fill(
    const int first_step, const std::vector<specfem::plugin::Buffer> &buffers) {
  if (scratch_.empty() || buffers.empty())
    return;

  // One step at a time, so the scratch stays at points x components whatever
  // the window size.
  const int nsteps = buffers.front().number_of_steps();
  for (std::size_t g = 0; g < groups_.size(); ++g) {
    const Group &group = groups_[g];
    const int count = group.sampler.number_of_points();
    if (count == 0)
      continue;
    const std::span<type_real> step_values(
        scratch_.data(), static_cast<std::size_t>(count) *
                             fk_injection::number_of_sample_components);
    for (int s = 0; s < nsteps; ++s) {
      group.sampler.sample(first_step + s, 1, step_values);
      for (const auto &slot : group.slots) {
        const int offset = sample_offset(slot.quantity);
        for (int p = 0; p < count; ++p) {
          const std::size_t row = static_cast<std::size_t>(p) *
                                  fk_injection::number_of_sample_components;
          for (int c = 0; c < slot.count; ++c)
            buffers[g].set(p, slot.offset + c, s,
                           static_cast<double>(step_values[row + offset + c]));
        }
      }
    }
  }
}
