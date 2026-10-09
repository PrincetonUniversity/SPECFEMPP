#include "specfem/injection/fk/layered_model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

specfem::injection::fk::LayeredModel::LayeredModel(
    std::vector<AcousticLayer> fluid_layers,
    std::vector<ElasticIsotropicLayer> elastic_layers)
    : fluid_layers_(std::move(fluid_layers)),
      elastic_layers_(std::move(elastic_layers)) {
  // Compute cumulative top depths in global order: fluid layers first, then
  // elastic layers.
  const int n_fluid = static_cast<int>(fluid_layers_.size());
  const int n_elastic = static_cast<int>(elastic_layers_.size());
  const int total = n_fluid + n_elastic;
  top_depths_.resize(total, static_cast<type_real>(0));

  for (int i = 1; i < n_fluid; ++i) {
    top_depths_[i] = top_depths_[i - 1] + fluid_layers_[i - 1].thickness;
  }
  type_real elastic_base =
      (n_fluid > 0)
          ? top_depths_[n_fluid - 1] + fluid_layers_[n_fluid - 1].thickness
          : static_cast<type_real>(0);
  if (n_fluid > 0) {
    top_depths_[n_fluid] = elastic_base;
  }
  for (int i = 1; i < n_elastic; ++i) {
    top_depths_[n_fluid + i] =
        top_depths_[n_fluid + i - 1] + elastic_layers_[i - 1].thickness;
  }

  validate();
}

const std::vector<specfem::injection::fk::AcousticLayer> &
specfem::injection::fk::LayeredModel::fluid_layers() const {
  return fluid_layers_;
}

const std::vector<specfem::injection::fk::ElasticIsotropicLayer> &
specfem::injection::fk::LayeredModel::elastic_layers() const {
  return elastic_layers_;
}

int specfem::injection::fk::LayeredModel::number_of_fluid_layers() const {
  return static_cast<int>(fluid_layers_.size());
}

int specfem::injection::fk::LayeredModel::number_of_elastic_layers() const {
  return static_cast<int>(elastic_layers_.size());
}

int specfem::injection::fk::LayeredModel::total_number_of_layers() const {
  return number_of_fluid_layers() + number_of_elastic_layers();
}

bool specfem::injection::fk::LayeredModel::has_fluid_layer() const {
  return number_of_fluid_layers() > 0;
}

type_real
specfem::injection::fk::LayeredModel::shear_modulus(int elastic_index) const {
  return elastic_layers_[elastic_index].density *
         elastic_layers_[elastic_index].s_velocity *
         elastic_layers_[elastic_index].s_velocity;
}

const std::vector<type_real> &
specfem::injection::fk::LayeredModel::layer_top_depths() const {
  return top_depths_;
}

double specfem::injection::fk::LayeredModel::fluid_thickness() const {
  double total = 0.0;
  for (const AcousticLayer &layer : fluid_layers_) {
    total += static_cast<double>(layer.thickness);
  }
  return total;
}

double specfem::injection::fk::LayeredModel::minimum_velocity() const {
  double min_vel = std::numeric_limits<double>::max();
  for (const AcousticLayer &layer : fluid_layers_) {
    min_vel = std::min(min_vel, static_cast<double>(layer.p_velocity));
  }
  for (const ElasticIsotropicLayer &layer : elastic_layers_) {
    min_vel = std::min(min_vel, static_cast<double>(layer.s_velocity));
  }
  return min_vel;
}

specfem::injection::fk::LayerLocation
specfem::injection::fk::LayeredModel::locate(double depth,
                                             point_medium requested,
                                             double tolerance) const {
  if (depth < -tolerance) {
    throw std::runtime_error(
        "LayeredModel::locate: depth above the free surface");
  }

  const int n_fluid = number_of_fluid_layers();
  const int n_elastic = number_of_elastic_layers();
  const double ft = fluid_thickness();

  // Whether depth falls exactly on the fluid-solid interface.
  const bool at_fluid_solid_iface =
      has_fluid_layer() && (std::fabs(depth - ft) <= tolerance);

  // Resolve medium.
  point_medium medium;
  double effective_depth = depth;

  if (requested == point_medium::from_depth) {
    // Acoustic only if inside the fluid column (strictly above the interface).
    if (has_fluid_layer() && depth < ft - tolerance) {
      medium = point_medium::acoustic;
    } else {
      medium = point_medium::elastic;
    }
  } else if (requested == point_medium::acoustic) {
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
    medium = point_medium::acoustic;
  } else { // requested == point_medium::elastic
    if (has_fluid_layer() && depth < ft - tolerance) {
      throw std::runtime_error(
          "LayeredModel::locate: elastic medium requested inside the fluid "
          "column");
    }
    if (at_fluid_solid_iface) {
      // Snap to the top of the elastic column.
      effective_depth = ft;
    }
    medium = point_medium::elastic;
  }

  // Locate the layer within the resolved block (first-match / tie → upper).
  int layer_index = 0;
  double height = 0.0;

  if (medium == point_medium::acoustic) {
    for (int i = 0; i < n_fluid; ++i) {
      const double top_i = static_cast<double>(top_depths_[i]);
      const double bottom_i =
          top_i + static_cast<double>(fluid_layers_[i].thickness);
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
      const double top_j = static_cast<double>(top_depths_[n_fluid + j]);
      const double bottom_j =
          top_j + static_cast<double>(elastic_layers_[j].thickness);
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

  const bool in_halfspace =
      (medium == point_medium::elastic && layer_index == n_elastic - 1);

  // Half-space top depth is always the last elastic layer's top depth.
  const double hs_top =
      static_cast<double>(top_depths_[n_fluid + n_elastic - 1]);
  const double depth_below_halfspace_top = hs_top - effective_depth;

  const bool at_free_surface =
      (medium == point_medium::acoustic && layer_index == 0 &&
       effective_depth <= tolerance);

  return LayerLocation{ medium,
                        layer_index,
                        height,
                        in_halfspace,
                        depth_below_halfspace_top,
                        at_free_surface };
}

void specfem::injection::fk::LayeredModel::validate() const {
  if (elastic_layers_.empty()) {
    throw std::runtime_error(
        "LayeredModel: elastic_layers must be non-empty (need a half-space).");
  }

  for (int i = 0; i < static_cast<int>(fluid_layers_.size()); ++i) {
    const AcousticLayer &layer = fluid_layers_[i];
    if (layer.density <= static_cast<type_real>(0)) {
      std::ostringstream msg;
      msg << "LayeredModel: fluid layer " << i << " has non-positive density ("
          << layer.density << ").";
      throw std::runtime_error(msg.str());
    }
    if (layer.p_velocity <= static_cast<type_real>(0)) {
      std::ostringstream msg;
      msg << "LayeredModel: fluid layer " << i
          << " has non-positive p_velocity (" << layer.p_velocity << ").";
      throw std::runtime_error(msg.str());
    }
  }

  for (int i = 0; i < static_cast<int>(elastic_layers_.size()); ++i) {
    const ElasticIsotropicLayer &layer = elastic_layers_[i];
    if (layer.density <= static_cast<type_real>(0)) {
      std::ostringstream msg;
      msg << "LayeredModel: elastic layer " << i
          << " has non-positive density (" << layer.density << ").";
      throw std::runtime_error(msg.str());
    }
    if (layer.p_velocity <= static_cast<type_real>(0)) {
      std::ostringstream msg;
      msg << "LayeredModel: elastic layer " << i
          << " has non-positive p_velocity (" << layer.p_velocity << ").";
      throw std::runtime_error(msg.str());
    }
    if (layer.s_velocity <= static_cast<type_real>(0)) {
      std::ostringstream msg;
      msg << "LayeredModel: elastic layer " << i
          << " has non-positive s_velocity (" << layer.s_velocity
          << "). Elastic layers require s_velocity > 0.";
      throw std::runtime_error(msg.str());
    }
  }
}
