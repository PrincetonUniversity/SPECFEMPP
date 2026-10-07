#include "specfem/globe_model/evaluator.hpp"

#include <stdexcept>

namespace specfem::globe_model {

bool Evaluator::is_active_ = false;

Dims Evaluator::dims() {
  throw std::runtime_error(
      "This build cannot evaluate globe model properties; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

Scales Evaluator::scales() {
  throw std::runtime_error(
      "This build cannot evaluate globe model properties; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

bool Evaluator::is_active() { return is_active_; }

Evaluator::Evaluator(const ModelConfig &, const std::string &) {
  throw std::runtime_error(
      "This build cannot consume globe meshes; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

Evaluator::~Evaluator() = default;

ElementProperties
Evaluator::evaluate_element(int, int, double, double, bool, bool,
                            const std::vector<double> &) const {
  throw std::runtime_error(
      "This build cannot evaluate globe model properties; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

ReferencePoint prem_reference(double, int, int) {
  throw std::runtime_error(
      "This build cannot evaluate PREM reference properties; configure with "
      "SPECFEM_BUILD_MESHFEM3D_GLOBE=ON");
}

} // namespace specfem::globe_model
