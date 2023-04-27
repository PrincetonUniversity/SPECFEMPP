#ifndef _DOMAIN_HPP
#define _DOMAIN_HPP

#include "compute/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

// using simd_type = Kokkos::Experimental::native_simd<double>;

namespace specfem {
namespace Domain {

/**
 * @brief  Base Domain class
 *
 */
class Domain {

public:
  /**
   * @brief Get a view of the field stored on the device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  virtual specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_field() const {
    return this->field;
  }
  /**
   * @brief Get a view of the field stored on the host
   *
   * @return specfem::kokkos::HostMirror2d<type_real>
   */
  virtual specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_field() const {
    return this->h_field;
  }
  /**
   * @brief Get a view of the derivative of field stored on the device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  virtual specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_field_dot() const {
    return this->field_dot;
  }
  /**
   * @brief Get a view of the derivative of field stored on the host
   *
   * @return specfem::kokkos::HostMirror2d<type_real>
   */
  virtual specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_field_dot() const {
    return this->h_field_dot;
  }
  /**
   * @brief Get a view of the second derivative of field stored on the Device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  virtual specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_field_dot_dot() const {
    return this->field_dot_dot;
  }
  /**
   * @brief Get a view of the second derivative of field stored on the host
   *
   * @return specfem::kokkos::HostMirror2d<type_real>
   */
  virtual specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_field_dot_dot() const {
    return this->h_field_dot_dot;
  }
  /**
   * @brief Get a view of rmass_inverse stored on the device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  virtual specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_rmass_inverse() const {
    return this->rmass_inverse;
  }
  /**
   * @brief Get a view of rmass_inverse stored on the host
   *
   * @return specfem::kokkos::HostMirror2d<type_real>
   */
  virtual specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_rmass_inverse() const {
    return this->h_rmass_inverse;
  }
  /**
   * @brief Compute interaction of stiffness matrix on second derivative of
   * field
   *
   */
  virtual void compute_stiffness_interaction_calling_routine(){};
  /**
   * @brief Divide the second derivative of field by the mass matrix
   *
   */
  virtual void divide_mass_matrix(){};
  /**
   * @brief Compute interaction of sources on second derivative of field
   *
   * @param timeval
   */
  virtual void compute_source_interaction(const type_real timeval){};

  /**
   * @brief Sync field views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  virtual void sync_field(specfem::sync::kind kind){};
  /**
   * @brief Sync derivative of field views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  virtual void sync_field_dot(specfem::sync::kind kind){};
  /**
   * @brief Sync second derivative of field views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  virtual void sync_field_dot_dot(specfem::sync::kind kind){};
  /**
   * @brief Sync inverse of mass matrix views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  virtual void sync_rmass_inverse(specfem::sync::kind kind){};
  /**
   * @brief Compute seismograms at for all receivers at isig_step
   *
   * @param seismogram_types DeviceView of types of seismograms to be calculated
   * @param isig_step timestep for seismogram calculation
   */
  virtual void compute_seismogram(const int isig_step){};
  // /**
  //  * @brief Load arrays required for compute forces into simd_arrays when
  //  * compiled with explicit SIMD types, or else reference original arrays.
  //  *
  //  */
  // virtual void simd_configure_arrays();

private:
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
      field; ///< View of field on Device
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
      h_field; ///< View of field on host
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
      field_dot; ///< View of derivative of
                 ///< field on Device
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
      h_field_dot; ///< View of derivative
                   ///< of field on host
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
      field_dot_dot; ///< View of second
                     ///< derivative of
                     ///< field on Device
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
      h_field_dot_dot; ///< View of second
                       ///< derivative of
                       ///< field on host
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
      rmass_inverse; ///< View of inverse
                     ///< of mass matrix on
                     ///< device
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
      h_rmass_inverse; ///< View of inverse
                       ///< of mass matrix
                       ///< on host
};

} // namespace Domain
} // namespace specfem

#endif
