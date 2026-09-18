#include "specfem/io/globe_model.hpp"

#include <stdexcept>
#include <utility>

namespace specfem::io {

bool globe_model::is_active_ = false;

globe_model::Dimensions globe_model::dimensions() {
  throw std::runtime_error(
      "This build cannot evaluate globe model properties; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

globe_model::Scales globe_model::query_scales() {
  throw std::runtime_error(
      "This build cannot evaluate globe model properties; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

bool globe_model::is_active() noexcept { return is_active_; }

globe_model::globe_model(const GlobeModelConfig &,
                         const specfem::globe::PlanetConstants &,
                         const std::string &) {
  throw std::runtime_error(
      "This build cannot consume globe meshes; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

globe_model::~globe_model() = default;

globe_model::globe_model(globe_model &&other) noexcept
    : constants_(std::move(other.constants_)), scales_(other.scales_),
      owns_state_(other.owns_state_) {
  other.owns_state_ = false;
}

globe_model &globe_model::operator=(globe_model &&other) noexcept {
  if (this != &other) {
    constants_ = std::move(other.constants_);
    scales_ = other.scales_;
    owns_state_ = other.owns_state_;
    other.owns_state_ = false;
  }
  return *this;
}

void globe_model::release() noexcept {}

specfem::globe::PlanetConstants::Radii globe_model::radii() const {
  throw std::runtime_error(
      "This build cannot query globe model radii; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

globe_model::ElementProperties
globe_model::evaluate_element(int, int, double, double, bool, bool,
                              const std::vector<double> &) const {
  throw std::runtime_error(
      "This build cannot evaluate globe model properties; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

globe_model::ReferencePoint globe_model::prem_reference(double, int,
                                                        int) const {
  throw std::runtime_error(
      "This build cannot evaluate PREM reference properties; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

} // namespace specfem::io
