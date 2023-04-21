#ifndef DOMAIN_H
#define DOMAIN_H

#include "../include/compute.h"
#include "../include/config.h"
#include "../include/quadrature.h"
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

/**
 * @brief Elastic domain class
 *
 * Elastic domains implementation details:
 *  - field -> Displacements along the 2 dimensions (idim) for every global
 * point (iglob) stored as a 2D View field(iglob, idim)
 *  - field_dot -> Velocity along the 2 dimensions (idim) for every global point
 * (iglob) stored as a 2D View field(iglob, idim)
 *  - field_dot_dot -> Acceleration along the 2 dimensions (idim) for every
 * global point (iglob) stored as a 2D View field(iglob, idim)
 */
class Elastic : public Domain {
public:
  /**
   * @brief Get a view of displacement stored on the device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_field() const override {
    return this->field;
  }
  /**
   * @brief Get a view of displacement stored on the hsot
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_field() const override {
    return this->h_field;
  }
  /**
   * @brief Get a view of velocity stored on device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_field_dot() const override {
    return this->field_dot;
  }
  /**
   * @brief Get a view of velocity stored on host
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_field_dot() const override {
    return this->h_field_dot;
  }
  /**
   * @brief Get a view of acceleration stored on device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_field_dot_dot() const override {
    return this->field_dot_dot;
  }
  /**
   * @brief Get a view of acceleration stored on host
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_field_dot_dot() const override {
    return this->h_field_dot_dot;
  }
  /**
   * @brief Get a view of inverse of mass matrix stored on device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_rmass_inverse() const override {
    return this->rmass_inverse;
  }
  /**
   * @brief Get a view of inverse of mass matrix stored on host
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_rmass_inverse() const override {
    return this->h_rmass_inverse;
  }

  /**
   * @brief Construct a new Elastic domain object
   *
   * This contructor helps in instantiating fields. Without instantiating any
   * material or mesh related private fields
   *
   * @param ndim Number of dimensions
   * @param nglob Total number of distinct quadrature points
   */
  Elastic(const int ndim, const int nglob);

  /**
   * @brief Construct a new Elastic domain object
   *
   * @param ndim Number of dimensions
   * @param nglob Total number of distinct quadrature points inside the
   * domain
   * @param compute Pointer to specfem::compute::compute struct
   * @param material_properties Pointer to specfem::compute::properties
   * struct
   * @param partial_derivatives Pointer to
   * specfem::compute::partial_derivatives struct
   * @param sources Pointer to specfem::compute::sources struct
   * @param quadx Pointer to quadrature object in x-dimension
   * @param quadx Pointer to quadrature object in z-dimension
   */
  Elastic(const int ndim, const int nglob, specfem::compute::compute *compute,
          specfem::compute::properties *material_properties,
          specfem::compute::partial_derivatives *partial_derivatives,
          specfem::compute::sources *sources,
          specfem::compute::receivers *receivers,
          specfem::quadrature::quadrature *quadx,
          specfem::quadrature::quadrature *quadz);
  /**
   * @brief Compute interaction of stiffness matrix on acceleration
   *
   */
  void compute_stiffness_interaction_calling_routine() override;
  template <int NGLL> void compute_stiffness_interaction();
  void compute_stiffness_interaction();

  /**
   * @brief Divide the acceleration by the mass matrix
   *
   */
  void divide_mass_matrix() override;
  /**
   * @brief Compute interaction of sources on acceleration
   *
   * @param timeval
   */
  void compute_source_interaction(const type_real timeval) override;
  /**
   * @brief Sync displacements views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  void sync_field(specfem::sync::kind kind) override;
  /**
   * @brief Sync velocity views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  void sync_field_dot(specfem::sync::kind kind) override;
  /**
   * @brief Sync acceleration views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  void sync_field_dot_dot(specfem::sync::kind kind) override;
  /**
   * @brief Sync inverse of mass matrix views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  void sync_rmass_inverse(specfem::sync::kind kind) override;
  /**
   * @brief function used to assign host and device views used in elastic domain
   * class
   *
   */
  KOKKOS_IMPL_HOST_FUNCTION
  void assign_views();
  /**
   * @brief Compute seismograms at for all receivers at isig_step
   *
   * @param seismogram_types DeviceView of types of seismograms to be calculated
   * @param isig_step timestep for seismogram calculation
   */
  void compute_seismogram(const int isig_step) override;
  // /**
  //  * @brief Load arrays required for compute forces into simd_arrays when
  //  * compiled with explicit SIMD types, or else reference original arrays.
  //  *
  //  */
  // void simd_configure_arrays() override;

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
      h_rmass_inverse;                ///< View of inverse
                                      ///< of mass matrix
                                      ///< on host
  specfem::compute::compute *compute; ///< Pointer to compute struct used to
                                      ///< store spectral element numbering
                                      ///< mapping (ibool)
  specfem::compute::properties *material_properties; ///< Pointer to struct used
                                                     ///< to store material
                                                     ///< properties
  specfem::compute::partial_derivatives *partial_derivatives; ///< Pointer to
                                                              ///< struct used
                                                              ///< to store
                                                              ///< partial
                                                              ///< derivates
  specfem::compute::sources *sources;     ///< Pointer to struct used to store
                                          ///< sources
  specfem::compute::receivers *receivers; ///< Pointer to struct used to store
                                          ///< receivers
  quadrature::quadrature *quadx;          ///< Pointer to quadrature object in
                                          ///< x-dimension
  quadrature::quadrature *quadz;          ///< Pointer to quadrature object in
                                          ///< z-dimension
  int nelem_domain; ///< Total number of elements in this domain
  specfem::kokkos::DeviceView1d<int> ispec_domain; ///< Array containing global
                                                   ///< indices(ispec) of all
                                                   ///< elements in this domain
                                                   ///< on the device
  specfem::kokkos::HostMirror1d<int> h_ispec_domain; ///< Array containing
                                                     ///< global indices(ispec)
                                                     ///< of all elements in
                                                     ///< this domain on the
                                                     ///< host
};
} // namespace Domain
} // namespace specfem

#endif
