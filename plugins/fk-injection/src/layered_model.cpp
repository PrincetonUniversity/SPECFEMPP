#include "fk_injection/layered_model.hpp"
#include <cmath>
#include <limits>
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
      if (first_elastic_layer_ < i) {
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
      first_elastic_layer_ = i + 1;
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
      depth += static_cast<double>(solid.thickness);
    }
  }
}

double fk_injection::LayeredModel::fluid_thickness() const {
  return top_depths_[first_elastic_layer_];
}

double fk_injection::LayeredModel::halfspace_top_depth() const {
  return top_depths_.back();
}

fk_injection::LayerLocation
fk_injection::LayeredModel::locate(const double depth,
                                   const specfem::plugin::Medium medium,
                                   const double tolerance) const {
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

  const int n_fluid = first_elastic_layer_;
  const double ft = fluid_thickness();

  if (medium == specfem::plugin::Medium::acoustic) {
    if (n_fluid == 0) {
      throw std::runtime_error(
          "LayeredModel::locate: acoustic medium requested but model has no "
          "fluid layer");
    }
    if (depth > ft + tolerance) {
      throw std::runtime_error(
          "LayeredModel::locate: acoustic medium requested below the fluid "
          "column");
    }
  } else if (n_fluid > 0 && depth < ft - tolerance) {
    throw std::runtime_error(
        "LayeredModel::locate: elastic medium requested inside the fluid "
        "column");
  }
  // First match, so an interface goes to the upper layer.
  const int halfspace = number_of_layers() - 1;
  const int first = medium == specfem::plugin::Medium::acoustic ? 0 : n_fluid;
  const int last =
      medium == specfem::plugin::Medium::acoustic ? n_fluid - 1 : halfspace;
  int layer = last;
  for (int i = first; i < last; ++i) {
    if (depth <= top_depths_[i + 1] + tolerance) {
      layer = i;
      break;
    }
  }
  return locate_in(layer, depth, tolerance);
}

fk_injection::LayerLocation
fk_injection::LayeredModel::locate_in(const int layer, const double depth,
                                      const double tolerance) const {
  if (!(tolerance >= 0.0)) {
    throw std::runtime_error(
        "LayeredModel::locate_in: tolerance must be non-negative");
  }
  const int halfspace = number_of_layers() - 1;
  if (layer < 0 || layer > halfspace) {
    std::ostringstream message;
    message << "LayeredModel::locate_in: layer " << layer << " outside [0, "
            << halfspace << "]";
    throw std::runtime_error(message.str());
  }
  const double top = top_depths_[layer];
  const double bottom = layer < halfspace
                            ? top_depths_[layer + 1]
                            : std::numeric_limits<double>::infinity();
  if (!(depth >= top - tolerance && depth <= bottom + tolerance)) {
    std::ostringstream message;
    message << "LayeredModel::locate_in: depth " << depth << " lies outside "
            << "layer " << layer << " (" << top << " to " << bottom << ")";
    throw std::runtime_error(message.str());
  }

  double snapped = depth;
  if (std::fabs(depth - top) <= tolerance)
    snapped = top;
  else if (std::fabs(depth - bottom) <= tolerance)
    snapped = bottom;

  const specfem::plugin::Medium medium =
      std::holds_alternative<fk_injection::AcousticLayer>(layers_[layer])
          ? specfem::plugin::Medium::acoustic
          : specfem::plugin::Medium::elastic;
  const bool in_halfspace = layer == halfspace;
  return LayerLocation{ medium,
                        layer,
                        in_halfspace ? 0.0 : bottom - snapped,
                        in_halfspace,
                        top_depths_.back() - snapped,
                        medium == specfem::plugin::Medium::acoustic &&
                            layer == 0 && snapped == 0.0,
                        depth };
}
