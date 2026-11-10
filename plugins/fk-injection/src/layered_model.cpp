#include "fk_injection/layered_model.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

fk_injection::LayeredModel::LayeredModel(
    std::vector<AcousticLayer> fluid_layers,
    std::vector<ElasticIsotropicLayer> elastic_layers)
    : fluid_layers_(std::move(fluid_layers)),
      elastic_layers_(std::move(elastic_layers)) {
  validate();

  top_depths_.reserve(fluid_layers_.size() + elastic_layers_.size());
  double depth = 0.0;
  for (const AcousticLayer &layer : fluid_layers_) {
    top_depths_.push_back(depth);
    depth += static_cast<double>(layer.thickness);
  }
  for (const ElasticIsotropicLayer &layer : elastic_layers_) {
    top_depths_.push_back(depth);
    depth += static_cast<double>(layer.thickness);
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

fk_injection::LayerLocation fk_injection::LayeredModel::locate(
    double depth, std::optional<specfem::plugin::Medium> requested,
    double tolerance) const {
  if (!(tolerance >= 0.0)) {
    throw std::runtime_error(
        "LayeredModel::locate: tolerance must be non-negative");
  }
  if (!(depth >= -tolerance)) {
    throw std::runtime_error(
        "LayeredModel::locate: depth above the free surface");
  }
  if (requested == specfem::plugin::Medium::elastic_spin) {
    throw std::runtime_error(
        "LayeredModel::locate: FK supports acoustic and elastic media only");
  }

  const int n_fluid = number_of_fluid_layers();
  const int n_elastic = number_of_elastic_layers();
  const double ft = fluid_thickness();

  // Whether depth falls exactly on the fluid-solid interface.
  const bool at_fluid_solid_iface =
      has_fluid_layer() && (std::fabs(depth - ft) <= tolerance);

  // Resolve medium.
  specfem::plugin::Medium medium = specfem::plugin::Medium::elastic;
  double effective_depth = depth;

  if (!requested) {
    // Acoustic only if inside the fluid column (strictly above the interface).
    if (has_fluid_layer() && depth < ft - tolerance) {
      medium = specfem::plugin::Medium::acoustic;
    }
  } else if (*requested == specfem::plugin::Medium::acoustic) {
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
    if (at_fluid_solid_iface) {
      // Snap to the bottom of the fluid column.
      effective_depth = ft;
    }
    medium = specfem::plugin::Medium::acoustic;
  } else { // *requested == specfem::plugin::Medium::elastic
    if (has_fluid_layer() && depth < ft - tolerance) {
      throw std::runtime_error(
          "LayeredModel::locate: elastic medium requested inside the fluid "
          "column");
    }
    if (at_fluid_solid_iface) {
      // Snap to the top of the elastic column.
      effective_depth = ft;
    }
  }

  // Locate the layer within the resolved block (first-match / tie → upper).
  int layer_index = 0;
  double height = 0.0;

  if (medium == specfem::plugin::Medium::acoustic) {
    for (int i = 0; i < n_fluid; ++i) {
      const double bottom_i = top_depths_[i + 1];
      if (effective_depth <= bottom_i + tolerance) {
        layer_index = i;
        height = bottom_i - effective_depth;
        break;
      }
    }
  } else { // elastic
    for (int j = 0; j < n_elastic; ++j) {
      const bool is_halfspace = (j == n_elastic - 1);
      if (is_halfspace) {
        layer_index = j;
        height = 0.0;
        break;
      }
      const double bottom_j = top_depths_[n_fluid + j + 1];
      if (effective_depth <= bottom_j + tolerance) {
        layer_index = j;
        height = bottom_j - effective_depth;
        break;
      }
    }
  }

  // Clamp small negative height produced by tolerance slack.
  if (height < 0.0) {
    height = 0.0;
  }

  const bool in_halfspace = (medium == specfem::plugin::Medium::elastic &&
                             layer_index == n_elastic - 1);

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
                        at_free_surface };
}

void fk_injection::LayeredModel::validate() const {
  if (elastic_layers_.empty()) {
    throw std::runtime_error(
        "LayeredModel: elastic_layers must be non-empty (need a half-space).");
  }

  // The negated comparisons also reject NaN.
  for (int i = 0; i < static_cast<int>(fluid_layers_.size()); ++i) {
    const AcousticLayer &layer = fluid_layers_[i];
    if (!(layer.density > static_cast<type_real>(0))) {
      std::ostringstream msg;
      msg << "LayeredModel: fluid layer " << i << " has non-positive density ("
          << layer.density << ").";
      throw std::runtime_error(msg.str());
    }
    if (!(layer.p_velocity > static_cast<type_real>(0))) {
      std::ostringstream msg;
      msg << "LayeredModel: fluid layer " << i
          << " has non-positive p_velocity (" << layer.p_velocity << ").";
      throw std::runtime_error(msg.str());
    }
    if (!(layer.thickness > static_cast<type_real>(0))) {
      std::ostringstream msg;
      msg << "LayeredModel: fluid layer " << i
          << " has non-positive thickness (" << layer.thickness << ").";
      throw std::runtime_error(msg.str());
    }
  }

  const int halfspace_index = static_cast<int>(elastic_layers_.size()) - 1;
  for (int i = 0; i <= halfspace_index; ++i) {
    const ElasticIsotropicLayer &layer = elastic_layers_[i];
    if (!(layer.density > static_cast<type_real>(0))) {
      std::ostringstream msg;
      msg << "LayeredModel: elastic layer " << i
          << " has non-positive density (" << layer.density << ").";
      throw std::runtime_error(msg.str());
    }
    if (!(layer.p_velocity > static_cast<type_real>(0))) {
      std::ostringstream msg;
      msg << "LayeredModel: elastic layer " << i
          << " has non-positive p_velocity (" << layer.p_velocity << ").";
      throw std::runtime_error(msg.str());
    }
    if (!(layer.s_velocity > static_cast<type_real>(0))) {
      std::ostringstream msg;
      msg << "LayeredModel: elastic layer " << i
          << " has non-positive s_velocity (" << layer.s_velocity
          << "). Elastic layers require s_velocity > 0.";
      throw std::runtime_error(msg.str());
    }
    if (i < halfspace_index && !(layer.thickness > static_cast<type_real>(0))) {
      std::ostringstream msg;
      msg << "LayeredModel: elastic layer " << i
          << " has non-positive thickness (" << layer.thickness << ").";
      throw std::runtime_error(msg.str());
    }
  }
}
