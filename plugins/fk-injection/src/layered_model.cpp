#include "fk_injection/layered_model.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace fk_injection::layered_model_impl {

/// Rejects a non-positive property; the negated comparison also rejects NaN.
void require_positive(type_real value, int layer, const char *medium,
                      const char *property);

} // namespace fk_injection::layered_model_impl

void fk_injection::layered_model_impl::require_positive(const type_real value,
                                                        const int layer,
                                                        const char *medium,
                                                        const char *property) {
  if (value > static_cast<type_real>(0))
    return;
  std::ostringstream message;
  message << "LayeredModel: layer " << layer << " (" << medium
          << ") has non-positive " << property << " (" << value << ").";
  throw std::runtime_error(message.str());
}

fk_injection::LayeredModel::LayeredModel(
    std::vector<fk_injection::Layer> layers)
    : layers_(std::move(layers)) {
  if (layers_.empty() ||
      !std::holds_alternative<fk_injection::ElasticIsotropicLayer>(
          layers_.back())) {
    throw std::runtime_error(
        "LayeredModel: the last layer must be an elastic half-space.");
  }

  const int halfspace = number_of_layers() - 1;
  top_depths_.reserve(layers_.size());
  double depth = 0.0;
  for (int i = 0; i <= halfspace; ++i) {
    top_depths_.push_back(depth);
    if (const auto *fluid =
            std::get_if<fk_injection::AcousticLayer>(&layers_[i])) {
      if (!elastic_layers_.empty()) {
        std::ostringstream message;
        message << "LayeredModel: acoustic layer " << i
                << " lies below an elastic layer; FK supports a fluid block "
                   "on top of the solid only.";
        throw std::runtime_error(message.str());
      }
      fk_injection::layered_model_impl::require_positive(fluid->density, i,
                                                         "acoustic", "density");
      fk_injection::layered_model_impl::require_positive(
          fluid->p_velocity, i, "acoustic", "p_velocity");
      fk_injection::layered_model_impl::require_positive(
          fluid->thickness, i, "acoustic", "thickness");
      fluid_layers_.push_back(*fluid);
      depth += static_cast<double>(fluid->thickness);
    } else {
      const auto &solid =
          std::get<fk_injection::ElasticIsotropicLayer>(layers_[i]);
      fk_injection::layered_model_impl::require_positive(solid.density, i,
                                                         "elastic", "density");
      fk_injection::layered_model_impl::require_positive(
          solid.p_velocity, i, "elastic", "p_velocity");
      fk_injection::layered_model_impl::require_positive(
          solid.s_velocity, i, "elastic", "s_velocity");
      if (i < halfspace)
        fk_injection::layered_model_impl::require_positive(
            solid.thickness, i, "elastic", "thickness");
      elastic_layers_.push_back(solid);
      depth += static_cast<double>(solid.thickness);
    }
  }
}

const std::vector<fk_injection::AcousticLayer> &
fk_injection::LayeredModel::fluid_layers() const {
  return fluid_layers_;
}

const std::vector<fk_injection::ElasticIsotropicLayer> &
fk_injection::LayeredModel::elastic_layers() const {
  return elastic_layers_;
}

int fk_injection::LayeredModel::number_of_fluid_layers() const {
  return static_cast<int>(fluid_layers_.size());
}

int fk_injection::LayeredModel::number_of_elastic_layers() const {
  return static_cast<int>(elastic_layers_.size());
}

bool fk_injection::LayeredModel::has_fluid_layer() const {
  return number_of_fluid_layers() > 0;
}

double fk_injection::LayeredModel::fluid_thickness() const {
  return top_depths_[fluid_layers_.size()];
}

double fk_injection::LayeredModel::halfspace_top_depth() const {
  return top_depths_.back();
}

fk_injection::LayerLocation
fk_injection::LayeredModel::locate(double depth,
                                   const specfem::plugin::Medium medium,
                                   double tolerance) const {
  if (!(tolerance >= 0.0)) {
    throw std::runtime_error(
        "LayeredModel::locate: tolerance must be non-negative");
  }
  if (!(depth >= -tolerance)) {
    throw std::runtime_error(
        "LayeredModel::locate: depth above the free surface");
  }
  if (medium != specfem::plugin::Medium::acoustic &&
      medium != specfem::plugin::Medium::elastic) {
    throw std::runtime_error(
        "LayeredModel::locate: FK supports acoustic and elastic media only");
  }

  const int n_fluid = number_of_fluid_layers();
  const double ft = fluid_thickness();

  // Whether depth falls exactly on the fluid-solid interface.
  const bool at_fluid_solid_iface =
      has_fluid_layer() && (std::fabs(depth - ft) <= tolerance);

  double effective_depth = depth;
  if (medium == specfem::plugin::Medium::acoustic) {
    if (!has_fluid_layer()) {
      throw std::runtime_error(
          "LayeredModel::locate: acoustic medium requested but model has no "
          "fluid layer");
    }
    if (depth > ft + tolerance) {
      throw std::runtime_error(
          "LayeredModel::locate: acoustic medium requested below the fluid "
          "column");
    }
  } else if (has_fluid_layer() && depth < ft - tolerance) {
    throw std::runtime_error(
        "LayeredModel::locate: elastic medium requested inside the fluid "
        "column");
  }
  // Snap to the fluid-solid interface.
  if (at_fluid_solid_iface)
    effective_depth = ft;

  // First match, so an interface goes to the upper layer.
  const int first = medium == specfem::plugin::Medium::acoustic ? 0 : n_fluid;
  const int last = medium == specfem::plugin::Medium::acoustic
                       ? n_fluid - 1
                       : number_of_layers() - 1;
  int layer_index = last;
  double height = 0.0;
  for (int i = first; i < number_of_layers() - 1 && i <= last; ++i) {
    const double bottom = top_depths_[i + 1];
    if (effective_depth <= bottom + tolerance) {
      layer_index = i;
      height = bottom - effective_depth;
      break;
    }
  }

  // Clamp small negative height produced by tolerance slack.
  if (height < 0.0) {
    height = 0.0;
  }

  const bool in_halfspace = layer_index == number_of_layers() - 1;

  // Half-space top depth is always the last elastic layer's top depth.
  const double depth_below_halfspace_top = top_depths_.back() - effective_depth;

  const bool at_free_surface =
      (medium == specfem::plugin::Medium::acoustic && layer_index == 0 &&
       effective_depth <= tolerance);

  return LayerLocation{ medium,
                        layer_index,
                        height,
                        in_halfspace,
                        depth_below_halfspace_top,
                        at_free_surface,
                        depth };
}
