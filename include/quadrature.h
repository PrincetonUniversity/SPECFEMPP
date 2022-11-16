#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace quadrature {

using DeviceView1d = specfem::DeviceView1d<type_real>;
using DeviceView2d = specfem::DeviceView2d<type_real>;
using HostMirror1d = specfem::HostMirror1d<type_real>;
using HostMirror2d = specfem::HostMirror2d<type_real>;

class quadrature {
  /**
   * @brief Defines the GLL/GLJ quadrature and related matrices required for
   * quadrature integration
   *
   */
public:
  /**
   * @brief Construct a new quadrature object
   *
   */
  quadrature();
  /**
   * @brief Construct a new quadrature object
   *
   * @param alpha alpha value for the polynomial functions
   * @param beta beta value for the polynomial functions
   *
   * @note if alpha = 0.0 and beta = 0.0 implies GLL quadrature. if alpha = 0.0
   * and beta = 1.0 implies GLJ quadrature
   */
  quadrature(const type_real alpha, const type_real beta);
  /**
   * @brief Construct a new quadrature object
   *
   * @param alpha alpha value for the polynomial functions
   * @param beta beta value for the polynomial functions
   * @param N Degree of the polynomial used
   *
   * @note if alpha = 0.0 and beta = 0.0 implies GLL quadrature. if alpha = 0.0
   * and beta = 1.0 implies GLJ quadrature
   */
  quadrature(const type_real alpha, const type_real beta, const int N);

  /**
   * @brief Define derivation matrices. Sets values of xi, w & hprime and
   * transfers the values to device.
   */
  void set_derivation_matrices();
  /**
   * @brief Get the value of xi on the device
   *
   * @return DeviceView1d xi
   */
  DeviceView1d get_xi() const;
  /**
   * @brief Get the w on the device
   *
   * @return DeviceView1d w
   */
  DeviceView1d get_w() const;
  /**
   * @brief Get the hprime on the device
   *
   * @return DeviceView2d hprime
   */
  DeviceView2d get_hprime() const;
  int get_N() const;

private:
  type_real alpha; //!< alpha value for the quadrature
  type_real beta;  //!< beta value for the quadrature
  int N;           //!< degree of the polynomial used for quadrature
  DeviceView1d xi, w;
  DeviceView2d hprime;
  HostMirror1d h_xi, h_w;
  HostMirror2d h_hprime;
  /**
   * @brief Set allocations for private variables
   *
   */
  void set_allocations();
  /**
   * @brief sync mirror and device views
   *
   */
  void sync_views();
};
} // namespace quadrature
