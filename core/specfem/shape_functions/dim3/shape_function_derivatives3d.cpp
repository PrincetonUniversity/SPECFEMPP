#include "shape_functions.hpp"
#include <stdexcept>
#include <vector>

namespace {
template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              bool>::type = true>
std::vector<std::vector<T> >
shape_function_derivatives_8node(const T xi, const T eta, const T zeta) {
  std::vector<std::vector<T> > dshape(3, std::vector<T>(8, 0.0));

  // d/dxi
  dshape[0][0] = -0.125 * (1.0 - eta) * (1.0 - zeta);
  dshape[0][1] = 0.125 * (1.0 - eta) * (1.0 - zeta);
  dshape[0][2] = 0.125 * (1.0 + eta) * (1.0 - zeta);
  dshape[0][3] = -0.125 * (1.0 + eta) * (1.0 - zeta);
  dshape[0][4] = -0.125 * (1.0 - eta) * (1.0 + zeta);
  dshape[0][5] = 0.125 * (1.0 - eta) * (1.0 + zeta);
  dshape[0][6] = 0.125 * (1.0 + eta) * (1.0 + zeta);
  dshape[0][7] = -0.125 * (1.0 + eta) * (1.0 + zeta);

  // d/deta
  dshape[1][0] = -0.125 * (1.0 - xi) * (1.0 - zeta);
  dshape[1][1] = -0.125 * (1.0 + xi) * (1.0 - zeta);
  dshape[1][2] = 0.125 * (1.0 + xi) * (1.0 - zeta);
  dshape[1][3] = 0.125 * (1.0 - xi) * (1.0 - zeta);
  dshape[1][4] = -0.125 * (1.0 - xi) * (1.0 + zeta);
  dshape[1][5] = -0.125 * (1.0 + xi) * (1.0 + zeta);
  dshape[1][6] = 0.125 * (1.0 + xi) * (1.0 + zeta);
  dshape[1][7] = 0.125 * (1.0 - xi) * (1.0 + zeta);

  // d/dzeta
  dshape[2][0] = -0.125 * (1.0 - xi) * (1.0 - eta);
  dshape[2][1] = -0.125 * (1.0 + xi) * (1.0 - eta);
  dshape[2][2] = -0.125 * (1.0 + xi) * (1.0 + eta);
  dshape[2][3] = -0.125 * (1.0 - xi) * (1.0 + eta);
  dshape[2][4] = 0.125 * (1.0 - xi) * (1.0 - eta);
  dshape[2][5] = 0.125 * (1.0 + xi) * (1.0 - eta);
  dshape[2][6] = 0.125 * (1.0 + xi) * (1.0 + eta);
  dshape[2][7] = 0.125 * (1.0 - xi) * (1.0 + eta);

  return dshape;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              bool>::type = true>
std::vector<std::vector<T> >
shape_function_derivatives_27node(const T xi, const T eta, const T zeta) {
  std::vector<std::vector<T> > dshape(3, std::vector<T>(27, 0.0));

  dshape[0][0] =
      (xi - 0.5) * (0.5 * eta * (eta - 1.0)) * (0.5 * zeta * (zeta - 1.0));
  dshape[0][1] =
      (xi + 0.5) * (0.5 * eta * (eta - 1.0)) * (0.5 * zeta * (zeta - 1.0));
  dshape[0][2] =
      (xi + 0.5) * (0.5 * eta * (eta + 1.0)) * (0.5 * zeta * (zeta - 1.0));
  dshape[0][3] =
      (xi - 0.5) * (0.5 * eta * (eta + 1.0)) * (0.5 * zeta * (zeta - 1.0));
  dshape[0][4] =
      (xi - 0.5) * (0.5 * eta * (eta - 1.0)) * (0.5 * zeta * (zeta + 1.0));
  dshape[0][5] =
      (xi + 0.5) * (0.5 * eta * (eta - 1.0)) * (0.5 * zeta * (zeta + 1.0));
  dshape[0][6] =
      (xi + 0.5) * (0.5 * eta * (eta + 1.0)) * (0.5 * zeta * (zeta + 1.0));
  dshape[0][7] =
      (xi - 0.5) * (0.5 * eta * (eta + 1.0)) * (0.5 * zeta * (zeta + 1.0));

  dshape[1][0] =
      (0.5 * xi * (xi - 1.0)) * (eta - 0.5) * (0.5 * zeta * (zeta - 1.0));
  dshape[1][1] =
      (0.5 * xi * (xi + 1.0)) * (eta - 0.5) * (0.5 * zeta * (zeta - 1.0));
  dshape[1][2] =
      (0.5 * xi * (xi + 1.0)) * (eta + 0.5) * (0.5 * zeta * (zeta - 1.0));
  dshape[1][3] =
      (0.5 * xi * (xi - 1.0)) * (eta + 0.5) * (0.5 * zeta * (zeta - 1.0));
  dshape[1][4] =
      (0.5 * xi * (xi - 1.0)) * (eta - 0.5) * (0.5 * zeta * (zeta + 1.0));
  dshape[1][5] =
      (0.5 * xi * (xi + 1.0)) * (eta - 0.5) * (0.5 * zeta * (zeta + 1.0));
  dshape[1][6] =
      (0.5 * xi * (xi + 1.0)) * (eta + 0.5) * (0.5 * zeta * (zeta + 1.0));
  dshape[1][7] =
      (0.5 * xi * (xi - 1.0)) * (eta + 0.5) * (0.5 * zeta * (zeta + 1.0));

  dshape[2][0] =
      (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta - 1.0)) * (zeta - 0.5);
  dshape[2][1] =
      (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta - 1.0)) * (zeta - 0.5);
  dshape[2][2] =
      (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta + 1.0)) * (zeta - 0.5);
  dshape[2][3] =
      (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta + 1.0)) * (zeta - 0.5);
  dshape[2][4] =
      (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta - 1.0)) * (zeta + 0.5);
  dshape[2][5] =
      (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta - 1.0)) * (zeta + 0.5);
  dshape[2][6] =
      (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta + 1.0)) * (zeta + 0.5);
  dshape[2][7] =
      (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta + 1.0)) * (zeta + 0.5);

  dshape[0][8] =
      (-2.0 * xi) * (0.5 * eta * (eta - 1.0)) * (0.5 * zeta * (zeta - 1.0));
  dshape[0][9] = (xi + 0.5) * (1.0 - eta * eta) * (0.5 * zeta * (zeta - 1.0));
  dshape[0][10] =
      (-2.0 * xi) * (0.5 * eta * (eta + 1.0)) * (0.5 * zeta * (zeta - 1.0));
  dshape[0][11] = (xi - 0.5) * (1.0 - eta * eta) * (0.5 * zeta * (zeta - 1.0));
  dshape[0][12] = (xi - 0.5) * (0.5 * eta * (eta - 1.0)) * (1.0 - zeta * zeta);
  dshape[0][13] = (xi + 0.5) * (0.5 * eta * (eta - 1.0)) * (1.0 - zeta * zeta);
  dshape[0][14] = (xi + 0.5) * (0.5 * eta * (eta + 1.0)) * (1.0 - zeta * zeta);
  dshape[0][15] = (xi - 0.5) * (0.5 * eta * (eta + 1.0)) * (1.0 - zeta * zeta);
  dshape[0][16] =
      (-2.0 * xi) * (0.5 * eta * (eta - 1.0)) * (0.5 * zeta * (zeta + 1.0));
  dshape[0][17] = (xi + 0.5) * (1.0 - eta * eta) * (0.5 * zeta * (zeta + 1.0));
  dshape[0][18] =
      (-2.0 * xi) * (0.5 * eta * (eta + 1.0)) * (0.5 * zeta * (zeta + 1.0));
  dshape[0][19] = (xi - 0.5) * (1.0 - eta * eta) * (0.5 * zeta * (zeta + 1.0));

  dshape[1][8] = (1.0 - xi * xi) * (eta - 0.5) * (0.5 * zeta * (zeta - 1.0));
  dshape[1][9] =
      (0.5 * xi * (xi + 1.0)) * (-2.0 * eta) * (0.5 * zeta * (zeta - 1.0));
  dshape[1][10] = (1.0 - xi * xi) * (eta + 0.5) * (0.5 * zeta * (zeta - 1.0));
  dshape[1][11] =
      (0.5 * xi * (xi - 1.0)) * (-2.0 * eta) * (0.5 * zeta * (zeta - 1.0));
  dshape[1][12] = (0.5 * xi * (xi - 1.0)) * (eta - 0.5) * (1.0 - zeta * zeta);
  dshape[1][13] = (0.5 * xi * (xi + 1.0)) * (eta - 0.5) * (1.0 - zeta * zeta);
  dshape[1][14] = (0.5 * xi * (xi + 1.0)) * (eta + 0.5) * (1.0 - zeta * zeta);
  dshape[1][15] = (0.5 * xi * (xi - 1.0)) * (eta + 0.5) * (1.0 - zeta * zeta);
  dshape[1][16] = (1.0 - xi * xi) * (eta - 0.5) * (0.5 * zeta * (zeta + 1.0));
  dshape[1][17] =
      (0.5 * xi * (xi + 1.0)) * (-2.0 * eta) * (0.5 * zeta * (zeta + 1.0));
  dshape[1][18] = (1.0 - xi * xi) * (eta + 0.5) * (0.5 * zeta * (zeta + 1.0));
  dshape[1][19] =
      (0.5 * xi * (xi - 1.0)) * (-2.0 * eta) * (0.5 * zeta * (zeta + 1.0));

  dshape[2][8] = (1.0 - xi * xi) * (0.5 * eta * (eta - 1.0)) * (zeta - 0.5);
  dshape[2][9] = (0.5 * xi * (xi + 1.0)) * (1.0 - eta * eta) * (zeta - 0.5);
  dshape[2][10] = (1.0 - xi * xi) * (0.5 * eta * (eta + 1.0)) * (zeta - 0.5);
  dshape[2][11] = (0.5 * xi * (xi - 1.0)) * (1.0 - eta * eta) * (zeta - 0.5);
  dshape[2][12] =
      (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta - 1.0)) * (-2.0 * zeta);
  dshape[2][13] =
      (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta - 1.0)) * (-2.0 * zeta);
  dshape[2][14] =
      (0.5 * xi * (xi + 1.0)) * (0.5 * eta * (eta + 1.0)) * (-2.0 * zeta);
  dshape[2][15] =
      (0.5 * xi * (xi - 1.0)) * (0.5 * eta * (eta + 1.0)) * (-2.0 * zeta);
  dshape[2][16] = (1.0 - xi * xi) * (0.5 * eta * (eta - 1.0)) * (zeta + 0.5);
  dshape[2][17] = (0.5 * xi * (xi + 1.0)) * (1.0 - eta * eta) * (zeta + 0.5);
  dshape[2][18] = (1.0 - xi * xi) * (0.5 * eta * (eta + 1.0)) * (zeta + 0.5);
  dshape[2][19] = (0.5 * xi * (xi - 1.0)) * (1.0 - eta * eta) * (zeta + 0.5);

  dshape[0][20] = (-2.0 * xi) * (1.0 - eta * eta) * (0.5 * zeta * (zeta - 1.0));
  dshape[0][21] = (-2.0 * xi) * (0.5 * eta * (eta - 1.0)) * (1.0 - zeta * zeta);
  dshape[0][22] = (xi + 0.5) * (1.0 - eta * eta) * (1.0 - zeta * zeta);
  dshape[0][23] = (-2.0 * xi) * (0.5 * eta * (eta + 1.0)) * (1.0 - zeta * zeta);
  dshape[0][24] = (xi - 0.5) * (1.0 - eta * eta) * (1.0 - zeta * zeta);
  dshape[0][25] = (-2.0 * xi) * (1.0 - eta * eta) * (0.5 * zeta * (zeta + 1.0));

  dshape[1][20] = (1.0 - xi * xi) * (-2.0 * eta) * (0.5 * zeta * (zeta - 1.0));
  dshape[1][21] = (1.0 - xi * xi) * (eta - 0.5) * (1.0 - zeta * zeta);
  dshape[1][22] = (0.5 * xi * (xi + 1.0)) * (-2.0 * eta) * (1.0 - zeta * zeta);
  dshape[1][23] = (1.0 - xi * xi) * (eta + 0.5) * (1.0 - zeta * zeta);
  dshape[1][24] = (0.5 * xi * (xi - 1.0)) * (-2.0 * eta) * (1.0 - zeta * zeta);
  dshape[1][25] = (1.0 - xi * xi) * (-2.0 * eta) * (0.5 * zeta * (zeta + 1.0));

  dshape[2][20] = (1.0 - xi * xi) * (1.0 - eta * eta) * (zeta - 0.5);
  dshape[2][21] = (1.0 - xi * xi) * (0.5 * eta * (eta - 1.0)) * (-2.0 * zeta);
  dshape[2][22] = (0.5 * xi * (xi + 1.0)) * (1.0 - eta * eta) * (-2.0 * zeta);
  dshape[2][23] = (1.0 - xi * xi) * (0.5 * eta * (eta + 1.0)) * (-2.0 * zeta);
  dshape[2][24] = (0.5 * xi * (xi - 1.0)) * (1.0 - eta * eta) * (-2.0 * zeta);
  dshape[2][25] = (1.0 - xi * xi) * (1.0 - eta * eta) * (zeta + 0.5);

  dshape[0][26] = (-2.0 * xi) * (1.0 - eta * eta) * (1.0 - zeta * zeta);
  dshape[1][26] = (1.0 - xi * xi) * (-2.0 * eta) * (1.0 - zeta * zeta);
  dshape[2][26] = (1.0 - xi * xi) * (1.0 - eta * eta) * (-2.0 * zeta);

  return dshape;
}
} // namespace

template <typename T>
std::vector<std::vector<T> >
specfem::shape_function::shape_function_derivatives(const T xi, const T eta,
                                                    const T zeta,
                                                    const int ngod) {

  static_assert(std::is_floating_point<T>::value,
                "Template type must be floating point");

  if (ngod == 8) {
    return shape_function_derivatives_8node(xi, eta, zeta);
  } else if (ngod == 27) {
    return shape_function_derivatives_27node(xi, eta, zeta);
  }
  throw std::invalid_argument("ngod must be either 8 or 27");
}

template std::vector<std::vector<float> >
specfem::shape_function::shape_function_derivatives(const float xi,
                                                    const float eta,
                                                    const float zeta,
                                                    const int ngod);

template std::vector<std::vector<double> >
specfem::shape_function::shape_function_derivatives(const double xi,
                                                    const double eta,
                                                    const double zeta,
                                                    const int ngod);
