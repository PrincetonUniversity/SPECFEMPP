#ifndef _PARAMETER_QUADRATURE_HPP
#define _PARAMETER_QUADRATURE_HPP

#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
namespace runtime_configuration {
/**
 * @brief Quadrature object is used to read and instantiate the
 * specfem::quadrature::quadrature classes in different dimensions
 *
 */
class quadrature {
public:
  /**
   * @brief Construct a new quadrature object
   *
   * @param alpha alpha value used to instantiate a
   * specfem::quadrature::quadrature class
   * @param beta beta value used to instantiate a
   * specfem::quadrature::quadrature class
   * @param ngllx number of quadrature points in x-dimension
   * @param ngllz number of quadrature points in z-dimension
   */
  quadrature(type_real alpha, type_real beta, int ngllx, int ngllz)
      : alpha(alpha), beta(beta), ngllx(ngllx), ngllz(ngllz){};
  /**
   * @brief Construct a new quadrature object
   *
   * @param Node YAML node describing the quadrature
   */
  quadrature(const YAML::Node &Node);
  /**
   * @brief Construct a new quadrature object
   *
   * @param quadrature pre-defined quadratures. e.g. GLL4 for 4th order GLL
   * quadrature
   */
  quadrature(const std::string quadrature);
  /**
   * @brief Instantiate quadrature objects in x and z dimensions
   *
   * @return std::tuple<specfem::quadrature::quadrature,
   * specfem::quadrature::quadrature> Quadrature objects in x and z dimensions
   */
  std::tuple<specfem::quadrature::quadrature *,
             specfem::quadrature::quadrature *>
  instantiate();

private:
  type_real alpha; ///< alpha value used to instantiate a
                   ///< specfem::quadrature::quadrature class
  type_real beta;  ///< beta value used to instantiate a
                   ///< specfem::quadrature::quadrature class
  int ngllx;       ///< number of quadrature points in x-dimension
  int ngllz;       ///< number of quadrature points in z-dimension
};

} // namespace runtime_configuration
} // namespace specfem

#endif
