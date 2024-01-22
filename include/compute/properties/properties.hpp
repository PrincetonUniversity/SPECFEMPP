#ifndef _COMPUTE_PROPERTIES_HPP
#define _COMPUTE_PROPERTIES_HPP

#include "enumerations/specfem_enums.hpp"
#include "impl/material_properties.hpp"
#include "impl/properties_container.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace specfem {
namespace compute {

/**
 * @brief Material properties stored at every quadrature point
 *
 */
struct properties {

  // template <specfem::enums::element::type type,
  //           specfem::enums::element::property_tag property>
  // struct material_property
  //     : public specfem::compute::impl::properties::properties_container<type,
  //     property> {
  //   constexpr static auto value_type = type;
  //   constexpr static auto property_type = property;

  //   material_property() = default;

  //   material_property(const int nspec, const int n_element, const int ngllz,
  //              const int ngllx, const specfem::mesh::materials &materials,
  //              const specfem::kokkos::HostView1d<int>
  //              property_index_mapping);
  // };

  int nspec; ///< total number of spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension
  specfem::kokkos::DeviceView1d<int> property_index_mapping;   ///< Mapping of
                                                               ///< spectral
                                                               ///< element to
                                                               ///< material
                                                               ///< properties
  specfem::kokkos::HostMirror1d<int> h_property_index_mapping; ///< Mapping of
                                                               ///< spectral
                                                               ///< element to
                                                               ///< material
                                                               ///< properties
  specfem::compute::impl::properties::material_property<
      specfem::enums::element::type::elastic,
      specfem::enums::element::property_tag::isotropic>
      elastic_isotropic;
  specfem::compute::impl::properties::material_property<
      specfem::enums::element::type::acoustic,
      specfem::enums::element::property_tag::isotropic>
      acoustic_isotropic;

  properties(const int nspec, const int ngllz, const int ngllx,
             const specfem::mesh::materials &materials);

  // /**
  //  * @name Material properties
  //  *
  //  * h_ prefixes denote views stored on host
  //  */
  // ///@{
  // specfem::kokkos::DeviceView3d<type_real> rho;
  // specfem::kokkos::HostMirror3d<type_real> h_rho;

  // specfem::kokkos::DeviceView3d<type_real> mu;
  // specfem::kokkos::HostMirror3d<type_real> h_mu;

  // specfem::kokkos::DeviceView3d<type_real> kappa;
  // specfem::kokkos::HostMirror3d<type_real> h_kappa;

  // specfem::kokkos::HostView3d<type_real> qmu;

  // specfem::kokkos::HostView3d<type_real> qkappa;

  // specfem::kokkos::HostView3d<type_real> rho_vp;

  // specfem::kokkos::HostView3d<type_real> rho_vs;

  // specfem::kokkos::DeviceView3d<type_real> rho_inverse;
  // specfem::kokkos::HostMirror3d<type_real> h_rho_inverse;

  // specfem::kokkos::DeviceView3d<type_real> lambdaplus2mu;
  // specfem::kokkos::HostMirror3d<type_real> h_lambdaplus2mu;

  // specfem::kokkos::DeviceView3d<type_real> lambdaplus2mu_inverse;
  // specfem::kokkos::HostMirror3d<type_real> h_lambdaplus2mu_inverse;

  // ///@}
  // // element type is defined in specfem_setup.hpp
  // specfem::kokkos::DeviceView1d<specfem::enums::element::type>
  //     ispec_type; ///< type of element
  //                 ///< stored on device
  // specfem::kokkos::HostMirror1d<specfem::enums::element::type>
  //     h_ispec_type; ///< type of element
  //                   ///< stored on host

  // /**
  //  * @brief Default constructor
  //  *
  //  */
  // properties(){};
  // /**
  //  * @brief Constructor to allocate views
  //  *
  //  * @param nspec Number for spectral elements
  //  * @param ngllz Number of quadrature points in z dimension
  //  * @param ngllx Number of quadrature points in x dimension
  //  */
  // properties(const int nspec, const int ngllz, const int ngllx);
  // /**
  //  * @brief Constructor to allocate and assign views
  //  *
  //  * @param kmato Material specification number
  //  * @param materials Pointer to material objects read from database file
  //  * @param nspec Number for spectral elements
  //  * @param ngllz Number of quadrature points in z dimension
  //  * @param ngllx Number of quadrature points in x dimension
  //  */
  // properties(const specfem::kokkos::HostView1d<int> kmato,
  //            const std::vector<std::shared_ptr<specfem::material::material> >
  //                &materials,
  //            const int nspec, const int ngllx, const int ngllz);

  // /**
  //  * @brief Helper routine to sync views within this struct
  //  *
  //  */
  // void sync_views();
};

// template <specfem::enums::element::type medium,
//           specfem::enums::element::property_tag property>
// struct element_properties {};

// template <>
// struct element_properties<specfem::enums::element::type::elastic,
//                           specfem::enums::element::property_tag::isotropic> {
//   type_real lambdaplus2mu;
//   type_real mu;
//   type_real rho;
//   type_real lambda;

//   type_real rho_vp;
//   type_real rho_vs;

//   KOKKOS_FUNCTION
//   element_properties() = default;

//   KOKKOS_FUNCTION
//   element_properties(const type_real &lambdaplus2mu, const type_real &mu,
//                      const type_real &rho)
//       : lambdaplus2mu(lambdaplus2mu), mu(mu), rho(rho),
//         lambda(lambdaplus2mu - 2 * mu), rho_vp(sqrt(rho * lambdaplus2mu)),
//         rho_vs(sqrt(rho * mu)) {}
// };

// template <>
// struct element_properties<specfem::enums::element::type::acoustic,
//                           specfem::enums::element::property_tag::isotropic> {
//   type_real lambdaplus2mu_inverse;
//   type_real rho_inverse;

//   type_real rho_vpinverse;

//   KOKKOS_FUNCTION
//   element_properties() = default;

//   KOKKOS_FUNCTION
//   element_properties(const type_real &lambdaplus2mu_inverse,
//                      const type_real &rho_inverse)
//       : lambdaplus2mu_inverse(lambdaplus2mu_inverse),
//       rho_inverse(rho_inverse),
//         rho_vpinverse(sqrt(rho_inverse * lambdaplus2mu_inverse)) {}
// };

} // namespace compute
} // namespace specfem

#endif
