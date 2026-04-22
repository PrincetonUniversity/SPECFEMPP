#include "specfem/utilities/logarithmic_center.hpp"
#include "specfem/setup.hpp"
#include <cmath>

type_real specfem::utilities::logarithmic_center(const type_real f1,
                                                 const type_real f2) {
  // use the logarithmic central frequency
  return std::pow(10.0, 0.5 * (std::log10(f1) + std::log10(f2)));
}
