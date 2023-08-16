#ifndef _ELASTIC_DOMAIN_HPP
#define _ELASTIC_DOMAIN_HPP

#include "compute/interface.hpp"
#include "domain/impl/elements/interface.hpp"
#include "domain/impl/receivers/interface.hpp"
#include "domain/impl/sources/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {

/**
 * @brief Specialized domain class for elastic media
 *
 * @tparam qp_type class used to define the quadrature points either at compile
 * time or run time
 */
template <typename qp_type>
class domain<specfem::enums::element::medium::elastic, qp_type> {

public:
  using dimension = specfem::enums::element::dimension::dim2;
  /**
   * @brief Get a view of field stored on the device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_field() const {
    return this->field;
  }
  /**
   * @brief Get a view of field stored on the host
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_field() const {
    return this->h_field;
  }
  /**
   * @brief Get a view of derivate of field stored on device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_field_dot() const {
    return this->field_dot;
  }
  /**
   * @brief Get a view of derivative of field stored on host
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_field_dot() const {
    return this->h_field_dot;
  }
  /**
   * @brief Get a view of double derivative of field stored on device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_field_dot_dot() const {
    return this->field_dot_dot;
  }
  /**
   * @brief Get a view of double derivative of field stored on host
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_field_dot_dot() const {
    return this->h_field_dot_dot;
  }
  /**
   * @brief Get a view of inverse of mass matrix stored on device
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  get_rmass_inverse() const {
    return this->rmass_inverse;
  }
  /**
   * @brief Get a view of inverse of mass matrix stored on host
   *
   * @return specfem::kokkos::DeviceView2d<type_real>
   */
  specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  get_host_rmass_inverse() const {
    return this->h_rmass_inverse;
  }

  /**
   * @brief Construct a new domain object
   *
   * @param ndim Number of dimensions
   * @param nglob Total number of distinct quadrature points
   * @param quadrature_points quadrature points to define compile time
   * quadrature or runtime quadrature
   * @param compute Compute struct used to store global index of each quadrature
   * point
   * @param material_properties Struct used to store material properties at each
   * quadrature point
   * @param partial_derivatives Struct used to store partial derivatives of
   * shape functions at each quadrature
   * @param compute_sources Struct used to store pre-computed source arrays at
   * each quadrature point and source time function
   * @param receivers Struct used to store pre-computed receiver arrays at each
   * quadrature point
   * @param quadx quadrature class in x direction
   * @param quadz quadrature class in z direction
   */
  domain(const int ndim, const int nglob, const qp_type &quadrature_points,
         specfem::compute::compute *compute,
         specfem::compute::properties material_properties,
         specfem::compute::partial_derivatives partial_derivatives,
         specfem::compute::sources compute_sources,
         specfem::compute::receivers compute_receivers,
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
  void divide_mass_matrix();
  /**
   * @brief Compute interaction of sources on acceleration
   *
   * @param timeval
   */
  void compute_source_interaction(const type_real timeval);
  /**
   * @brief Sync displacements views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  void sync_field(specfem::sync::kind kind);
  /**
   * @brief Sync velocity views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  void sync_field_dot(specfem::sync::kind kind);
  /**
   * @brief Sync acceleration views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  void sync_field_dot_dot(specfem::sync::kind kind);
  /**
   * @brief Sync inverse of mass matrix views between host and device
   *
   * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
   */
  void sync_rmass_inverse(specfem::sync::kind kind);
  /**
   * @brief Compute seismograms at for all receivers at isig_step
   *
   * @param seismogram_types DeviceView of types of seismograms to be
   * calculated
   * @param isig_step timestep for seismogram calculation
   */
  void compute_seismogram(const int isig_step);

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
  quadrature::quadrature *quadx;      ///< Pointer to quadrature object in
                                      ///< x-dimension
  quadrature::quadrature *quadz;      ///< Pointer to quadrature object in
                                      ///< z-dimension
  int nelem_domain; ///< Total number of elements in this domain
  specfem::kokkos::DeviceView1d<specfem::domain::impl::elements::container<
      specfem::domain::impl::elements::element<
          specfem::enums::element::dimension::dim2,
          specfem::enums::element::medium::elastic,
          qp_type> > >
      elements; ///< Container to store pointer to every element inside
                ///< this domain
  specfem::kokkos::DeviceView1d<specfem::domain::impl::sources::container<
      specfem::domain::impl::sources::source<
          specfem::enums::element::dimension::dim2,
          specfem::enums::element::medium::elastic,
          qp_type> > >
      sources; ///< Container to store pointer to every source inside
               ///< this domain
  specfem::kokkos::DeviceView1d<specfem::domain::impl::receivers::container<
      specfem::domain::impl::receivers::receiver<
          specfem::enums::element::dimension::dim2,
          specfem::enums::element::medium::elastic,
          qp_type> > >
      receivers; ///< Container to store pointer to every receiver inside
                 ///< this domain

  qp_type quadrature_points; ///< Quadrature points to define compile time
                             ///< quadrature or runtime quadrature
};

} // namespace domain
} // namespace specfem

#endif
