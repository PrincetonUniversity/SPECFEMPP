#pragma once

#include "config.hpp"
#include "plane_wave/plane_wave.hpp"
#include "specfem/plugin/injection_plugin.hpp"

#include <string>
#include <vector>

namespace plane_wave {

/**
 * @brief Injection plugin serving an analytic plane wave at elastic isotropic
 * points.
 *
 * Provides displacement, velocity, acceleration and traction. At setup it
 * checks that every point's material matches the configured whole space, so
 * the analytic field is the exact incident field of the mesh.
 */
class PlaneWavePlugin : public specfem::plugin::InjectionPlugin {
public:
  /**
   * @brief Read the configuration and prepare the plane wave.
   * @param config_path Path of the plugin's YAML configuration
   * @param info Simulation parameters
   * @param communicator MPI communicator (unused: every rank evaluates its
   * own points)
   * @throws specfem::plugin::Unsupported for a run that is not 3D
   */
  PlaneWavePlugin(const std::string &config_path,
                  const specfem::plugin::SimulationInfo &info,
                  const specfem::plugin::Communicator &communicator);

  /**
   * @brief Construct from an already parsed configuration.
   * @param config Configuration
   * @param info Simulation parameters
   * @throws specfem::plugin::Unsupported for a run that is not 3D
   */
  PlaneWavePlugin(const Config &config,
                  const specfem::plugin::SimulationInfo &info);

  bool supports(specfem::plugin::Medium medium,
                specfem::plugin::Property property,
                specfem::plugin::Quantity quantity) const override;

  /**
   * @brief Copy the points and check their material.
   * @param points Injection points
   * @throws specfem::plugin::Unsupported naming the first point whose
   * material differs from the configuration
   */
  void setup(const specfem::plugin::Points &points) override;

  void fill(int first_step,
            const std::vector<specfem::plugin::Buffer> &buffers) override;

private:
  /// What fill needs from one group of points
  struct Group {
    std::vector<Vector> positions;                    ///< Point coordinates
    std::vector<Vector> normals;                      ///< Unit normals
    std::vector<specfem::plugin::QuantitySlot> slots; ///< Requested quantities
  };

  void check_material(const specfem::plugin::Points &points,
                      const specfem::plugin::PointGroup &group) const;

  Config config_;
  PlaneWave wave_;
  double t0_;
  double dt_;
  std::vector<Group> groups_;
};

} // namespace plane_wave
