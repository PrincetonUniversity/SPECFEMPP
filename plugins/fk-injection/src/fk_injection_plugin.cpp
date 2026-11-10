#include "fk_injection_plugin.hpp"
#include "fk_injection/solver.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

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
      zmax_(config.top(info.bounds()[5])),
      horizontal_bounds_{ info.bounds()[0], info.bounds()[1], info.bounds()[2],
                          info.bounds()[3] } {}

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
  fk_injection::EvalPoints eval_points;
  eval_points.horizontal_bounds = horizontal_bounds_;
  for (const auto &group : points.groups()) {
    Group copy;
    copy.first_point = eval_points.size();
    copy.slots = group.slots();
    if (!copy.slots.empty()) {
      copy.count = group.size();
      for (std::int64_t p = 0; p < group.size(); ++p) {
        const std::int64_t i = group.begin() + p;
        const double depth = zmax_ - points.z(i);
        const Material material = point_material(group, p);
        check_material(points, i, group.medium(), depth, material);

        eval_points.x.push_back(static_cast<type_real>(points.x(i)));
        eval_points.y.push_back(static_cast<type_real>(points.y(i)));
        eval_points.z.push_back(static_cast<type_real>(depth));
        eval_points.normal_x.push_back(static_cast<type_real>(points.nx(i)));
        eval_points.normal_y.push_back(static_cast<type_real>(points.ny(i)));
        eval_points.normal_z.push_back(static_cast<type_real>(points.nz(i)));
        eval_points.medium.push_back(group.medium());

        // The stress series use sigma_rr = xi1 sigma_zz - 4 xim dx, as in
        // SPECFEM3D. They read 0 on fluid points, which have no stress.
        double xi1 = 0;
        double xim = 0;
        double bulk = 0;
        if (group.medium() == specfem::plugin::Medium::elastic) {
          const double xi =
              (material.vs * material.vs) / (material.vp * material.vp);
          const double mu = material.rho * material.vs * material.vs;
          const double lambda =
              material.rho * material.vp * material.vp - 2.0 * mu;
          xi1 = 1.0 - 2.0 * xi;
          xim = (1.0 - xi) * mu;
          bulk = lambda / (2.0 * (lambda + mu));
        }
        eval_points.lame_ratio_xi1.push_back(static_cast<type_real>(xi1));
        eval_points.lame_ratio_xim.push_back(static_cast<type_real>(xim));
        eval_points.lame_ratio_bulk.push_back(static_cast<type_real>(bulk));
      }
    }
    groups_.push_back(std::move(copy));
  }

  // A rank without points has nothing to solve, and solve needs at least one.
  if (eval_points.size() > 0)
    result_ = fk_injection::solve(config_.model, wave_, window_, eval_points,
                                  fk_injection::field_derivative::velocity,
                                  config_.convention);
  scratch_.resize(static_cast<std::size_t>(eval_points.size()) *
                  fk_injection::number_of_sample_components);
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
fk_injection::FkInjectionPlugin::layer_material(
    const specfem::plugin::Medium medium, const int layer) const {
  if (medium == specfem::plugin::Medium::acoustic) {
    const auto &fluid = config_.model.fluid_layers()[layer];
    return { static_cast<double>(fluid.density),
             static_cast<double>(fluid.p_velocity), 0.0 };
  }
  const auto &elastic = config_.model.elastic_layers()[layer];
  return { static_cast<double>(elastic.density),
           static_cast<double>(elastic.p_velocity),
           static_cast<double>(elastic.s_velocity) };
}

void fk_injection::FkInjectionPlugin::check_material(
    const specfem::plugin::Points &points, const std::int64_t index,
    const specfem::plugin::Medium medium, const double depth,
    const Material &material) const {
  const auto describe_point = [&] {
    std::ostringstream message;
    message << "fk: point " << index << " at (" << points.x(index) << ", "
            << points.y(index) << ", " << points.z(index) << ")";
    return message.str();
  };

  const double locate_tolerance = std::max(1e-3, 1e-6 * std::abs(depth));
  fk_injection::LayerLocation location;
  try {
    location = config_.model.locate(depth, medium, locate_tolerance);
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

  const Material layer = layer_material(medium, location.layer);
  if (matches(layer))
    return;

  // A GLL point on a layer interface carries its own element's material, which
  // can be the lower layer's. The field is continuous there, so either serves.
  const int layers_in_block = medium == specfem::plugin::Medium::acoustic
                                  ? config_.model.number_of_fluid_layers()
                                  : config_.model.number_of_elastic_layers();
  if (location.height <= locate_tolerance &&
      location.layer + 1 < layers_in_block &&
      matches(layer_material(medium, location.layer + 1)))
    return;

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
  for (int s = 0; s < nsteps; ++s) {
    result_.sample(first_step + s, 1, std::span<type_real>(scratch_));
    for (std::size_t g = 0; g < groups_.size(); ++g) {
      const Group &group = groups_[g];
      for (const auto &slot : group.slots) {
        const int offset = sample_offset(slot.quantity);
        for (std::int64_t p = 0; p < group.count; ++p) {
          const std::int64_t row = (group.first_point + p) *
                                   fk_injection::number_of_sample_components;
          for (int c = 0; c < slot.count; ++c)
            buffers[g].set(p, slot.offset + c, s,
                           static_cast<double>(scratch_[row + offset + c]));
        }
      }
    }
  }
}
