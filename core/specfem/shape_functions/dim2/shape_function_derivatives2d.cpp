
#include "shape_functions.hpp"
#include <stdexcept>
#include <vector>

namespace {

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              int>::type = 0>
std::vector<std::vector<T> > shape_function_derivatives_4node(const T xi,
                                                              const T gamma) {
  std::vector<std::vector<T> > dershape2D(2, std::vector<T>(4, 0.0));

  const T sp = xi + 1;
  const T sm = xi - 1;
  const T tp = gamma + 1;
  const T tm = gamma - 1;
  dershape2D[0][0] = 0.25 * tm;
  dershape2D[0][1] = -0.25 * tm;
  dershape2D[0][2] = 0.25 * tp;
  dershape2D[0][3] = -0.25 * tp;
  dershape2D[1][0] = 0.25 * sm;
  dershape2D[1][1] = -0.25 * sp;
  dershape2D[1][2] = 0.25 * sp;
  dershape2D[1][3] = -0.25 * sm;
  return dershape2D;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              int>::type = 0>
std::vector<std::vector<T> > shape_function_derivatives_9node(const T xi,
                                                              const T gamma) {
  std::vector<std::vector<T> > dershape2D(2, std::vector<T>(9, 0.0));

  const T sp = xi + 1.0;
  const T sm = xi - 1.0;
  const T tp = gamma + 1.0;
  const T tm = gamma - 1.0;
  const T s2 = xi * 2.0;
  const T t2 = gamma * 2.0;
  const T ss = xi * xi;
  const T tt = gamma * gamma;
  const T st = xi * gamma;

  //----  corner nodes
  dershape2D[0][0] = 0.25 * tm * gamma * (s2 - 1.0);
  dershape2D[0][1] = 0.25 * tm * gamma * (s2 + 1.0);
  dershape2D[0][2] = 0.25 * tp * gamma * (s2 + 1.0);
  dershape2D[0][3] = 0.25 * tp * gamma * (s2 - 1.0);

  dershape2D[1][0] = 0.25 * sm * xi * (t2 - 1.0);
  dershape2D[1][1] = 0.25 * sp * xi * (t2 - 1.0);
  dershape2D[1][2] = 0.25 * sp * xi * (t2 + 1.0);
  dershape2D[1][3] = 0.25 * sm * xi * (t2 + 1.0);

  //----  midside nodes
  dershape2D[0][4] = -1.0 * st * tm;
  dershape2D[0][5] = 0.5 * (1.0 - tt) * (s2 + 1.0);
  dershape2D[0][6] = -1.0 * st * tp;
  dershape2D[0][7] = 0.5 * (1.0 - tt) * (s2 - 1.0);

  dershape2D[1][4] = 0.5 * (1.0 - ss) * (t2 - 1.0);
  dershape2D[1][5] = -1.0 * st * sp;
  dershape2D[1][6] = 0.5 * (1.0 - ss) * (t2 + 1.0);
  dershape2D[1][7] = -1.0 * st * sm;

  //----  center node
  dershape2D[0][8] = -1.0 * s2 * (1.0 - tt);
  dershape2D[1][8] = -1.0 * t2 * (1.0 - ss);

  return dershape2D;
}

} // namespace

template <typename T>
std::vector<std::vector<T> >
specfem::shape_function::shape_function_derivatives(const T xi, const T gamma,
                                                    const int ngnod) {
  static_assert(std::is_floating_point<T>::value,
                "Template parameter must be floating point");
  if (ngnod == 4) {
    return shape_function_derivatives_4node(xi, gamma);
  } else if (ngnod == 9) {
    return shape_function_derivatives_9node(xi, gamma);
  } else {
    throw std::invalid_argument("Unsupported number of nodes");
  }
}

template std::vector<std::vector<float> >
specfem::shape_function::shape_function_derivatives(const float xi,
                                                    const float gamma,
                                                    const int ngnod);

template std::vector<std::vector<double> >
specfem::shape_function::shape_function_derivatives(const double xi,
                                                    const double gamma,
                                                    const int ngnod);
