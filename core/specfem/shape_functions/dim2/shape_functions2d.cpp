#include "shape_functions.hpp"
#include <stdexcept>
#include <vector>

namespace {

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              int>::type = 0>
std::vector<T> shape_function_4node(const T xi, const T gamma) {
  std::vector<T> shape2D(4, 0.0);
  shape2D[0] = 0.25 * (xi - 1) * (gamma - 1);
  shape2D[1] = -0.25 * (xi + 1) * (gamma - 1);
  shape2D[2] = 0.25 * (xi + 1) * (gamma + 1);
  shape2D[3] = -0.25 * (xi - 1) * (gamma + 1);
  return shape2D;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              int>::type = 0>
std::vector<T> shape_function_9node(const T xi, const T gamma) {
  std::vector<T> shape2D(9, 0.0);
  shape2D[0] = 0.25 * (xi - 1) * xi * (gamma - 1) * gamma;
  shape2D[1] = 0.25 * (xi + 1) * xi * (gamma - 1) * gamma;
  shape2D[2] = 0.25 * (xi + 1) * xi * (gamma + 1) * gamma;
  shape2D[3] = 0.25 * (xi - 1) * xi * (gamma + 1) * gamma;
  shape2D[4] = 0.5 * (1 - xi * xi) * (gamma - 1) * gamma;
  shape2D[5] = 0.5 * (xi + 1) * xi * (1 - gamma * gamma);
  shape2D[6] = 0.5 * (1 - xi * xi) * (gamma + 1) * gamma;
  shape2D[7] = 0.5 * (xi - 1) * xi * (1 - gamma * gamma);
  shape2D[8] = (1 - xi * xi) * (1 - gamma * gamma);
  return shape2D;
}

} // namespace

template <typename T>
std::vector<T> specfem::shape_function::shape_function(const T xi,
                                                       const T gamma,
                                                       const int ngnod) {

  static_assert(std::is_floating_point<T>::value,
                "Floating point required as template parameter");

  if (ngnod == 4) {
    return shape_function_4node(xi, gamma);
  } else if (ngnod == 9) {
    return shape_function_9node(xi, gamma);
  } else {
    throw std::invalid_argument("Error: wrong number of control nodes");
  }
}

template std::vector<float>
specfem::shape_function::shape_function(const float xi, const float gamma,
                                        const int ngnod);

template std::vector<double>
specfem::shape_function::shape_function(const double xi, const double gamma,
                                        const int ngnod);
