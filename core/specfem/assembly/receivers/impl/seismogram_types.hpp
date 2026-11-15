#pragma once

#include "specfem/enums.hpp"
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace specfem::assembly::receivers_impl {

/**
 * @brief Validated, ordered list of the seismogram types a run records.
 *
 * Owns the one place that rejects an unsupported type and the type -> storage
 * slot map used by the seismogram kernel and the writer.
 */
class SeismogramTypes {
public:
  SeismogramTypes() = default;

  /**
   * @brief Validate @p types and assign each its storage slot in order.
   *
   * @param types Seismogram types to record
   * @throws std::runtime_error if a type is not displacement, velocity,
   * acceleration, pressure, rotation, intrinsic_rotation or curl
   */
  explicit SeismogramTypes(const std::vector<specfem::enums::wavefield> &types)
      : types_(types) {
    for (int index = 0; index < static_cast<int>(types_.size()); ++index) {
      const auto type = types_[index];
      if (type != specfem::enums::wavefield::displacement &&
          type != specfem::enums::wavefield::velocity &&
          type != specfem::enums::wavefield::acceleration &&
          type != specfem::enums::wavefield::pressure &&
          type != specfem::enums::wavefield::rotation &&
          type != specfem::enums::wavefield::intrinsic_rotation &&
          type != specfem::enums::wavefield::curl) {
        std::ostringstream message;
        message << "Error reading specfem receiver configuration.(" << __FILE__
                << ":" << __LINE__ << ")\n";
        message << "Unknown seismogram type: "
                << specfem::enums::to_string(type) << "\n";
        message
            << "Valid seismogram types are: displacement, velocity, "
            << "acceleration, pressure, rotation, intrinsic_rotation, curl.\n";
        message << "Please check your configuration file.\n";
        throw std::runtime_error(message.str());
      }
      index_by_type_[type] = index;
    }
  }

  /// @return Number of configured seismogram types
  int size() const { return static_cast<int>(types_.size()); }

  /**
   * @brief Storage slot of @p type.
   *
   * @param type Seismogram type
   * @return Slot index in configuration order
   * @throws std::runtime_error if @p type was not configured
   */
  int index_of(const specfem::enums::wavefield type) const {
    const auto found = index_by_type_.find(type);
    if (found == index_by_type_.end()) {
      std::ostringstream message;
      message << "Seismogram type " << specfem::enums::to_string(type)
              << " was not configured for this run.";
      throw std::runtime_error(message.str());
    }
    return found->second;
  }

  /// @return Seismogram type stored in slot @p index
  specfem::enums::wavefield type_at(const int index) const {
    return types_.at(index);
  }

  /// @return Configured seismogram types in slot order
  const std::vector<specfem::enums::wavefield> &types() const { return types_; }

private:
  std::vector<specfem::enums::wavefield> types_;
  std::unordered_map<specfem::enums::wavefield, int> index_by_type_;
};

} // namespace specfem::assembly::receivers_impl
