#ifndef _ELASTIC_DOMAIN_HPP
#define _ELASTIC_DOMAIN_HPP

#include "compute/interface.hpp"
#include "domain/elastic_domain/impl/operators/gradient2d.hpp"
#include "domain/elastic_domain/impl/operators/stress2d.hpp"
#include "domain/elastic_domain/impl/operators/update_acceleration2d.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
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
template <typename element_type>
class domain<specfem::enums::medium::elastic, element_type> {
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
   * @brief Construct a new elastic domain object
   *
   * This contructor helps in instantiating fields. Without instantiating any
   * material or mesh related private fields
   *
   * @param ndim Number of dimensions
   * @param nglob Total number of distinct quadrature points
   */
  domain(const int ndim, const int nglob);

  /**
   * @brief Construct a new elastic domain object
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
  domain(const int ndim, const int nglob, specfem::compute::compute *compute,
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
      h_rmass_inverse;                    ///< View of inverse
                                          ///< of mass matrix
                                          ///< on host
  specfem::compute::compute *compute;     ///< Pointer to compute struct used to
                                          ///< store spectral element numbering
                                          ///< mapping (ibool)
  specfem::compute::sources *sources;     ///< Pointer to struct used to store
                                          ///< sources
  specfem::compute::receivers *receivers; ///< Pointer to struct used to store
                                          ///< receivers
  quadrature::quadrature *quadx;          ///< Pointer to quadrature object in
                                          ///< x-dimension
  quadrature::quadrature *quadz;          ///< Pointer to quadrature object in
                                          ///< z-dimension
  int nelem_domain; ///< Total number of elements in this domain
  specfem::kokkos::DeviceView1d<element_type> elements;   ///< Container to
                                                          ///< store pointer to
                                                          ///< every element
                                                          ///< inside
  specfem::kokkos::HostMirror1d<element_type> h_elements; ///< Container to
                                                          ///< store pointer to
                                                          ///< every element
                                                          ///< inside
};

} // namespace domain
} // namespace specfem

#endif
