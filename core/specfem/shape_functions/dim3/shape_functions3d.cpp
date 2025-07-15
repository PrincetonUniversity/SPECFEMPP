#include "shape_functions.hpp"
#include <stdexcept>
#include <vector>

namespace {

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              bool>::type = true>
std::vector<T> shape_function_8node(const T xi, const T eta, const T zeta) {
  std::vector<T> shape(8, 0.0);

  shape[0] = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta);
  shape[1] = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta);
  shape[2] = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta);
  shape[3] = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta);
  shape[4] = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zeta);
  shape[5] = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zeta);
  shape[6] = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zeta);
  shape[7] = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zeta);

  return shape;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              bool>::type = true>
std::vector<T> shape_function_27node(const T xi, const T eta, const T zeta) {

  std::vector<T> shape3D(27, 0.0);

  // Corner nodes
  shape3D[0] = (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta - 1.0)) *
               (0.5 * zeta * (zeta - 1.0));
  shape3D[1] = (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta - 1.0)) *
               (0.5 * zeta * (zeta - 1.0));
  shape3D[2] = (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta + 1.0)) *
               (0.5 * zeta * (zeta - 1.0));
  shape3D[3] = (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta + 1.0)) *
               (0.5 * zeta * (zeta - 1.0));
  shape3D[4] = (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta - 1.0)) *
               (0.5 * zeta * (zeta + 1.0));
  shape3D[5] = (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta - 1.0)) *
               (0.5 * zeta * (zeta + 1.0));
  shape3D[6] = (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta + 1.0)) *
               (0.5 * zeta * (zeta + 1.0));
  shape3D[7] = (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta + 1.0)) *
               (0.5 * zeta * (zeta + 1.0));

  // Mid-edge nodes
  shape3D[8] = (1.0 - (xi * xi)) * (0.5 * eta * (eta - 1.0)) *
               (0.5 * zeta * (zeta - 1.0));
  shape3D[9] = (0.5 * xi * (xi + 1.0)) * (1.0 - (eta * eta)) *
               (0.5 * zeta * (zeta - 1.0));
  shape3D[10] = (1.0 - (xi * xi)) * (0.5 * eta * (eta + 1.0)) *
                (0.5 * zeta * (zeta - 1.0));
  shape3D[11] = (0.5 * xi * (xi - 1.0)) * (1.0 - (eta * eta)) *
                (0.5 * zeta * (zeta - 1.0));
  shape3D[12] = (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta - 1.0)) *
                (1.0 - (zeta * zeta));
  shape3D[13] = (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta - 1.0)) *
                (1.0 - (zeta * zeta));
  shape3D[14] = (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta + 1.0)) *
                (1.0 - (zeta * zeta));
  shape3D[15] = (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta + 1.0)) *
                (1.0 - (zeta * zeta));
  shape3D[16] = (1.0 - (xi * xi)) * (0.5 * eta * (eta - 1.0)) *
                (0.5 * zeta * (zeta + 1.0));
  shape3D[17] = (0.5 * xi * (xi + 1.0)) * (1.0 - (eta * eta)) *
                (0.5 * zeta * (zeta + 1.0));
  shape3D[18] = (1.0 - (xi * xi)) * (0.5 * eta * (eta + 1.0)) *
                (0.5 * zeta * (zeta + 1.0));
  shape3D[19] = (0.5 * xi * (xi - 1.0)) * (1.0 - (eta * eta)) *
                (0.5 * zeta * (zeta + 1.0));

  // Face center node
  shape3D[20] =
      (1.0 - (xi * xi)) * (1.0 - (eta * eta)) * (0.5 * zeta * (zeta - 1.0));
  shape3D[21] =
      (1.0 - (xi * xi)) * (0.5 * eta * (eta - 1.0)) * (1.0 - (zeta * zeta));
  shape3D[22] =
      (0.5 * xi * (xi + 1.0)) * (1.0 - (eta * eta)) * (1.0 - (zeta * zeta));
  shape3D[23] =
      (1.0 - (xi * xi)) * (0.5 * eta * (eta + 1.0)) * (1.0 - (zeta * zeta));
  shape3D[24] =
      (0.5 * xi * (xi - 1.0)) * (1.0 - (eta * eta)) * (1.0 - (zeta * zeta));
  shape3D[25] =
      (1.0 - (xi * xi)) * (1.0 - (eta * eta)) * (0.5 * zeta * (zeta + 1.0));

  // Center node
  shape3D[26] = (1.0 - (xi * xi)) * (1.0 - (eta * eta)) * (1.0 - (zeta * zeta));

  return shape3D;
}
} // namespace

template <typename T>
std::vector<T> specfem::shape_function::shape_function(const T xi, const T eta,
                                                       const T zeta,
                                                       const int ngnod) {
  static_assert(std::is_floating_point<T>::value,
                "Floating point required as template parameter");
  if (ngnod == 8) {
    return shape_function_8node(xi, eta, zeta);
  } else if (ngnod == 27) {
    return shape_function_27node(xi, eta, zeta);
  } else {
    throw std::invalid_argument("Error: wrong number of control nodes");
  }
}

template std::vector<float>
specfem::shape_function::shape_function(const float xi, const float eta,
                                        const float zeta, const int ngnod);

template std::vector<double>
specfem::shape_function::shape_function(const double xi, const double eta,
                                        const double zeta, const int ngnod);
