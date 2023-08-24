#ifndef _DOMAIN_HPP
#define _DOMAIN_HPP

#include "compute/interface.hpp"
#include "impl/elements/interface.hpp"
#include "impl/receivers/interface.hpp"
#include "impl/sources/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {

/**
 * @brief domain class
 *
 * Domain class serves as the driver used to compute the elemental kernels. For
 * example, method @c compute_stiffness_interaction is used to
 * implement Kokkos parallelization and loading memory to scratch spaces, which
 * are then used by the elemental implementation to update acceleration. The
 * goal the domain class is to provide a general Kokkos parallelization
 * framework which can be used by specialized elemental implementations. This
 * allows us to hide the Kokkos parallelization details from the end developer
 * when implementing new physics (i.e. specialized elements).
 *
 *
 * @tparam medium class defining the domain medium. Separate implementations
 * exist for elastic, acoustic or poroelastic media
 * @tparam quadrature_points class used to define the quadrature points either
 * at compile time or run time
 *
 * Domain implementation details:
 *  - field -> stores a 2 dimensional field along different components. The
 * components may vary based of medium type. For example Acoustic domain have
 * only 1 component i.e. potential, Elastic domain have 2 components i.e. X,Z
 */
template <class medium, class qp_type> class domain {
public:
  using dimension = specfem::enums::element::dimension::dim2; ///< Dimension of
                                                              ///< the domain
  using medium_type = medium; ///< Type of medium i.e. acoustic, elastic or
                              ///< poroelastic
  using quadrature_points_type = qp_type; ///< Type of quadrature points i.e.
                                          ///< static or dynamic

  /**
   * @brief Alias for scratch view type
   *
   * @code
   *    template <typename T, int N>
   *    using ScratchViewType = Kokkos::View<T **[N], Kokkos::LayoutRight,
   * specfem::kokkos::DeviceScratchMemorySpace,
   * Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
   * @endcode
   *
   * @tparam T type of scratch view
   * @tparam N Extent of scratch view in 3rd dimension
   */
  template <typename T, int N>
  using ScratchViewType =
      typename quadrature_points_type::template ScratchViewType<T, N>;

  /**
   * @brief Construct a new domain object
   *
   * @param nglob Total number of distinct quadrature points
   * @param quadrature_points Type of quadrature points - either static (number
   * of quadrature points defined at compile time) or dynamic (number of
   * quadrature points defined at runtime)
   * @param compute Pointer to compute struct used to store spectral element
   * numbering mapping (ibool)
   * @param material_properties properties struct used to store material
   * properties at each quadrature point
   * @param partial_derivatives partial derivatives struct used to store partial
   * derivatives of basis functions at each quadrature point
   * @param compute_sources sources struct used to store source information
   * @param compute_receivers receivers struct used to store receiver
   * information
   * @param quadx Quadrature object in x-dimension
   * @param quadz Quadrature object in z-dimension
   */
  domain(const int nglob, const qp_type &quadrature_points,
         specfem::compute::compute *compute,
         specfem::compute::properties material_properties,
         specfem::compute::partial_derivatives partial_derivatives,
         specfem::compute::sources compute_sources,
         specfem::compute::receivers compute_receivers,
         specfem::quadrature::quadrature *quadx,
         specfem::quadrature::quadrature *quadz);

  /**
   * @brief Destroy the domain object
   *
   */
  ~domain() = default;

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
      specfem::domain::impl::elements::element<dimension, medium_type,
                                               quadrature_points_type> > >
      elements; ///< Container to store pointer to every element inside
                ///< this domain
  specfem::kokkos::DeviceView1d<specfem::domain::impl::sources::container<
      specfem::domain::impl::sources::source<dimension, medium_type,
                                             quadrature_points_type> > >
      sources; ///< Container to store pointer to every source inside
               ///< this domain
  specfem::kokkos::DeviceView1d<specfem::domain::impl::receivers::container<
      specfem::domain::impl::receivers::receiver<dimension, medium_type,
                                                 quadrature_points_type> > >
      receivers; ///< Container to store pointer to every receiver inside
                 ///< this domain

  qp_type quadrature_points; ///< Quadrature points to define compile time
                             ///< quadrature or runtime quadrature
};
} // namespace domain

} // namespace specfem

#endif
