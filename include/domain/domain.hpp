#ifndef _DOMAIN_HPP
#define _DOMAIN_HPP

#include "compute/interface.hpp"
#include "impl/interface.hpp"
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
 * Domain class serves as the driver used to compute the elemental kernels. It
 * describes the parallelism and updates the wavefield at each time step.
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
   * @param assembly object containing the assembly information
   */
  domain(const specfem::compute::assembly &assembly,
         const quadrature_points_type &quadrature_points);

  /**
   * @brief Default destructor
   *
   */
  ~domain() = default;

  /**
   * @brief Initialize the domain
   *
   */
  template <specfem::enums::time_scheme::type time_scheme>
  void mass_time_contribution(const type_real dt) {
    kernels.template mass_time_contribution<time_scheme>(dt);
  };

  // /**
  //  * @brief Compute interaction of stiffness matrix on acceleration
  //  *
  //  */
  // void compute_stiffness_interaction() {
  //   kernels.compute_stiffness_interaction();
  // };

  /**
   * @brief Invert the mass matrix
   *
   */
  void invert_mass_matrix();

  /**
   * @brief Divide the acceleration by the mass matrix
   *
   */
  void divide_mass_matrix();

  // /**
  //  * @brief Compute interaction of sources on acceleration
  //  *
  //  * @param timeval
  //  */
  // void compute_source_interaction(const type_real timeval) {
  //   kernels.compute_source_interaction(timeval);
  // };

  // /**
  //  * @brief Sync displacements views between host and device
  //  *
  //  * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
  //  */
  // void sync_field(specfem::sync::kind kind);

  // /**
  //  * @brief Sync velocity views between host and device
  //  *
  //  * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
  //  */
  // void sync_field_dot(specfem::sync::kind kind);

  // /**
  //  * @brief Sync acceleration views between host and device
  //  *
  //  * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
  //  */
  // void sync_field_dot_dot(specfem::sync::kind kind);

  // /**
  //  * @brief Sync inverse of mass matrix views between host and device
  //  *
  //  * @param kind defines sync direction i.e. DeviceToHost or HostToDevice
  //  */
  // void sync_rmass_inverse(specfem::sync::kind kind);

  // /**
  //  * @brief Compute seismograms at for all receivers at isig_step
  //  *
  //  * @param seismogram_types DeviceView of types of seismograms to be
  //  * calculated
  //  * @param isig_step timestep for seismogram calculation
  //  */
  // void compute_seismogram(const int isig_step) {
  //   kernels.compute_seismograms(isig_step);
  // };

  // /**
  //  * @brief Get a view of field stored on the device
  //  *
  //  * @return specfem::kokkos::DeviceView2d<type_real>
  //  */
  // specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  // get_field() const {
  //   return this->field;
  // }

  // /**
  //  * @brief Get a view of field stored on the host
  //  *
  //  * @return specfem::kokkos::DeviceView2d<type_real>
  //  */
  // specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  // get_host_field() const {
  //   return this->h_field;
  // }

  // /**
  //  * @brief Get a view of derivate of field stored on device
  //  *
  //  * @return specfem::kokkos::DeviceView2d<type_real>
  //  */
  // specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  // get_field_dot() const {
  //   return this->field_dot;
  // }

  // /**
  //  * @brief Get a view of derivative of field stored on host
  //  *
  //  * @return specfem::kokkos::DeviceView2d<type_real>
  //  */
  // specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  // get_host_field_dot() const {
  //   return this->h_field_dot;
  // }

  // /**
  //  * @brief Get a view of double derivative of field stored on device
  //  *
  //  * @return specfem::kokkos::DeviceView2d<type_real>
  //  */
  // specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  // get_field_dot_dot() const {
  //   return this->field_dot_dot;
  // }

  // /**
  //  * @brief Get a view of double derivative of field stored on host
  //  *
  //  * @return specfem::kokkos::DeviceView2d<type_real>
  //  */
  // specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  // get_host_field_dot_dot() const {
  //   return this->h_field_dot_dot;
  // }

  // /**
  //  * @brief Get a view of inverse of mass matrix stored on device
  //  *
  //  * @return specfem::kokkos::DeviceView2d<type_real>
  //  */
  // specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
  // get_rmass_inverse() const {
  //   return this->rmass_inverse;
  // }

  // /**
  //  * @brief Get a view of inverse of mass matrix stored on host
  //  *
  //  * @return specfem::kokkos::DeviceView2d<type_real>
  //  */
  // specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>
  // get_host_rmass_inverse() const {
  //   return this->h_rmass_inverse;
  // }

private:
  specfem::compute::impl::field_impl<medium_type> field; ///< Field object
  specfem::domain::impl::kernels::kernels<medium_type, quadrature_points_type>
      kernels; ///< Kernels object used to compute elemental kernels
};
} // namespace domain

} // namespace specfem

#endif
