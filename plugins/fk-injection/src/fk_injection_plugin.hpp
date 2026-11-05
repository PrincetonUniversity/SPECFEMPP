#pragma once

#include "field_sampler.hpp"
#include "plugin_config.hpp"
#include "specfem/plugin/injection_plugin.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace fk_injection {

/**
 * @brief Injection plugin serving the FK field of a layered model.
 *
 * Provides velocity and traction at elastic isotropic points, displacement and
 * potential rate at acoustic isotropic points. At setup it checks that every
 * point's material matches the layer of the model at its depth, because the
 * model defines the injected field.
 */
class FkInjectionPlugin : public specfem::plugin::InjectionPlugin {
public:
  /**
   * @brief Read the configuration.
   * @param config_path Path of the plugin's YAML configuration
   * @param info Simulation parameters
   * @param communicator MPI communicator (unused: every rank solves for its
   * own points)
   */
  FkInjectionPlugin(const std::string &config_path,
                    const specfem::plugin::SimulationInfo &info,
                    const specfem::plugin::Communicator &communicator);

  /**
   * @brief Construct from an already parsed configuration.
   * @param config Configuration
   * @param info Simulation parameters
   */
  FkInjectionPlugin(const PluginConfig &config,
                    const specfem::plugin::SimulationInfo &info);

  bool supports(specfem::plugin::Medium medium,
                specfem::plugin::Property property,
                specfem::plugin::Quantity quantity) const override;

  /**
   * @brief Check the points' material, solve for the response of the model,
   * and synthesize the series the points need.
   *
   * The response depends on the mesh box only, so a rank without points
   * validates the configuration too.
   *
   * @param points Injection points
   * @throws specfem::plugin::Unsupported naming the first point whose material
   * differs from the model, or if the wave's ray parameter is out of range
   * @throws std::out_of_range if a group lacks a material property
   * @throws std::runtime_error if the time window doesn't cover the run
   */
  void setup(const specfem::plugin::Points &points) override;

  void fill(int first_step,
            const std::vector<specfem::plugin::Buffer> &buffers) override;

private:
  /// What fill needs from one group of points
  struct Group {
    FieldSampler sampler{}; ///< Field at the group's points; none without slots
    std::vector<specfem::plugin::QuantitySlot> slots; ///< Requested quantities
  };

  /// Material of a point; vs is 0 in a fluid
  struct Material {
    double rho = 0;
    double vp = 0;
    double vs = 0;
  };

  static Material point_material(const specfem::plugin::PointGroup &group,
                                 std::int64_t point);
  static int sample_offset(specfem::plugin::Quantity quantity);

  Material layer_material(specfem::plugin::Medium medium, int layer) const;
  void check_material(const specfem::plugin::Points &points, std::int64_t index,
                      specfem::plugin::Medium medium, double depth,
                      const Material &material) const;

  PluginConfig config_;
  IncidentWave wave_;
  TimeWindow window_;
  double zmax_;
  HorizontalBox box_; ///< Mesh box, shared by every rank
  std::vector<Group> groups_{};
  std::vector<type_real> scratch_{}; ///< One step of the largest group
};

} // namespace fk_injection
