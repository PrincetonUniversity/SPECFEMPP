#include "plane_wave_plugin.hpp"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>

plane_wave::PlaneWavePlugin::PlaneWavePlugin(
    const std::string &config_path, const specfem::plugin::SimulationInfo &info,
    const specfem::plugin::Communicator &)
    : PlaneWavePlugin(plane_wave::load_config(config_path), info) {}

plane_wave::PlaneWavePlugin::PlaneWavePlugin(
    const plane_wave::Config &config,
    const specfem::plugin::SimulationInfo &info)
    : config_(config), wave_(config.plane_wave()), t0_(info.t0()),
      dt_(info.dt()) {
  if (info.dimension() != 3)
    throw specfem::plugin::Unsupported(
        "plane-wave: only 3D simulations are supported, got dimension " +
        std::to_string(info.dimension()));
}

bool plane_wave::PlaneWavePlugin::supports(
    const specfem::plugin::Medium medium,
    const specfem::plugin::Property property,
    const specfem::plugin::Quantity quantity) const {
  if (medium != specfem::plugin::Medium::elastic ||
      property != specfem::plugin::Property::isotropic)
    return false;
  switch (quantity) {
  case specfem::plugin::Quantity::displacement:
  case specfem::plugin::Quantity::velocity:
  case specfem::plugin::Quantity::acceleration:
  case specfem::plugin::Quantity::traction:
    return true;
  default:
    return false;
  }
}

void plane_wave::PlaneWavePlugin::setup(const specfem::plugin::Points &points) {
  groups_.clear();
  for (const auto &group : points.groups()) {
    Group copy;
    copy.slots = group.slots();
    if (!copy.slots.empty() && group.size() > 0) {
      check_material(points, group);
      for (std::int64_t p = 0; p < group.size(); ++p) {
        const std::int64_t i = group.begin() + p;
        copy.positions.push_back({ points.x(i), points.y(i), points.z(i) });
        copy.normals.push_back({ points.nx(i), points.ny(i), points.nz(i) });
      }
    }
    groups_.push_back(std::move(copy));
  }
}

void plane_wave::PlaneWavePlugin::check_material(
    const specfem::plugin::Points &points,
    const specfem::plugin::PointGroup &group) const {
  const auto &expected = config_.material;
  for (std::int64_t p = 0; p < group.size(); ++p) {
    const double kappa = group.property(p, "kappa");
    const double mu = group.property(p, "mu");
    const double rho = group.property(p, "rho");
    const plane_wave::Material found{ rho,
                                      std::sqrt((kappa + 4 * mu / 3) / rho),
                                      std::sqrt(mu / rho) };

    const auto mismatch = [&](const char *name, const double want,
                              const double got) {
      if (std::abs(got - want) <= config_.material_tolerance * std::abs(want))
        return;
      const std::int64_t i = group.begin() + p;
      std::ostringstream message;
      message << "plane-wave: point " << i << " at (" << points.x(i) << ", "
              << points.y(i) << ", " << points.z(i) << ") has " << name << " "
              << got << ", but the configuration has " << want
              << " (relative tolerance " << config_.material_tolerance << ")";
      throw specfem::plugin::Unsupported(message.str());
    };
    mismatch("rho", expected.rho, found.rho);
    mismatch("vp", expected.vp, found.vp);
    mismatch("vs", expected.vs, found.vs);
  }
}

void plane_wave::PlaneWavePlugin::fill(
    const int first_step, const std::vector<specfem::plugin::Buffer> &buffers) {
  for (std::size_t g = 0; g < groups_.size(); ++g) {
    const Group &group = groups_[g];
    const specfem::plugin::Buffer &buffer = buffers[g];
    const auto npoints = static_cast<std::int64_t>(group.positions.size());
    const int nsteps = buffer.number_of_steps();

    // Each point writes only its own entries, so the result does not depend
    // on the number of threads.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t p = 0; p < npoints; ++p) {
      const plane_wave::Vector &x = group.positions[p];
      const plane_wave::Vector &n = group.normals[p];
      for (int s = 0; s < nsteps; ++s) {
        const double t = t0_ + (first_step + s) * dt_;
        for (const auto &slot : group.slots) {
          plane_wave::Vector value{};
          switch (slot.quantity) {
          case specfem::plugin::Quantity::displacement:
            value = wave_.displacement(x, t);
            break;
          case specfem::plugin::Quantity::velocity:
            value = wave_.velocity(x, t);
            break;
          case specfem::plugin::Quantity::acceleration:
            value = wave_.acceleration(x, t);
            break;
          case specfem::plugin::Quantity::traction:
            value = wave_.traction(x, n, t);
            break;
          default:
            break;
          }
          for (int c = 0; c < slot.count; ++c)
            buffer.set(p, slot.offset + c, s, value[c]);
        }
      }
    }
  }
}
