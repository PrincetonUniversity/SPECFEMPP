#include "specfem/injection/fk/layered_model.hpp"
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
