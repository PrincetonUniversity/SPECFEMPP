#include "../test_fixture/test_fixture.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "policies/chunk.hpp"
#include "specfem_setup.hpp"
#include <gtest/gtest.h>
/*

POROELASTIC PROPERTY NAMES
DEFINE_POINT_VALUE(phi, 0)        ///< porosity @f$ \phi @f$
DEFINE_POINT_VALUE(tortuosity, 1) ///< tortuosity @f$ \tau @f$
DEFINE_POINT_VALUE(rho_s, 2)      ///< solid density @f$ \rho_s @f$
DEFINE_POINT_VALUE(rho_f, 3)      ///< fluid density @f$ \rho_f @f$
DEFINE_POINT_VALUE(kappa_s, 4)    ///< solid bulk modulus @f$ \kappa_s @f$
DEFINE_POINT_VALUE(kappa_f, 5)    ///< fluid bulk modulus @f$ \kappa_f @f$
DEFINE_POINT_VALUE(kappa_fr, 6)   ///< frame bulk modulus @f$ \kappa_{fr} @f$
DEFINE_POINT_VALUE(mu_fr, 7)      ///< frame shear modulus @f$ \mu_{fr} @f$
DEFINE_POINT_VALUE(eta, 8)        ///< Viscosity @f$ \eta @f$
DEFINE_POINT_VALUE(Kxx, 9)        ///< permeability @f$ K_{xx} @f$
DEFINE_POINT_VALUE(Kzz, 10)       ///< permeability @f$ K_{zz} @f$
DEFINE_POINT_VALUE(Kxz, 11)       ///< permeability @f$ K_{xz} @f$

*/
// Template get_error_message
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd = false>
std::string get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2, MediumTag,
                                  PropertyTag, false> &point_kernel,
    const type_real value);

// Template get_error_message specialization: elastic isotropic (parent)
template <>
std::string get_error_message(
    const specfem::point::kernels<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, false> &point_kernel,
    const type_real value) {
  std::ostringstream message;

  message << "\n\t Expected: " << value;
  message << "\n\t Got: \n";
  message << "\t\trho = " << point_kernel.rho() << "\n";
  message << "\t\tmu = " << point_kernel.mu() << "\n";
  message << "\t\tkappa = " << point_kernel.kappa() << "\n";
  message << "\t\trhop = " << point_kernel.rhop() << "\n";
  message << "\t\talpha = " << point_kernel.alpha() << "\n";
  message << "\t\tbeta = " << point_kernel.beta() << "\n";

  return message.str();
}

// Template get_error_message specialization: elastic p-sv isotropic (child)
template <>
std::string get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_sv,
                                  specfem::element::property_tag::isotropic,
                                  false> &point_kernel,
    const type_real value) {

  return get_error_message(
      static_cast<specfem::point::kernels<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::property_tag::isotropic, false> >(point_kernel),
      value);
}

// Template get_error_message specialization: elastic sh isotropic (child)
template <>
std::string get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_sh,
                                  specfem::element::property_tag::isotropic,
                                  false> &point_kernel,
    const type_real value) {

  return get_error_message(
      static_cast<specfem::point::kernels<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::property_tag::isotropic, false> >(point_kernel),
      value);
}

// Template get_error_message specialization: elastic anisotropic (parent)
template <>
std::string get_error_message(
    const specfem::point::kernels<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, false> &point_kernel,
    const type_real value) {
  std::ostringstream message;

  message << "\n\t Expected: " << value;
  message << "\n\t Got: \n";
  message << "\t\trho = " << point_kernel.rho() << "\n";
  message << "\t\tc11 = " << point_kernel.c11() << "\n";
  message << "\t\tc13 = " << point_kernel.c13() << "\n";
  message << "\t\tc15 = " << point_kernel.c15() << "\n";
  message << "\t\tc33 = " << point_kernel.c33() << "\n";
  message << "\t\tc35 = " << point_kernel.c35() << "\n";
  message << "\t\tc55 = " << point_kernel.c55() << "\n";

  return message.str();
}

// Template get_error_message specialization: elastic p-sv anisotropic (child)
template <>
std::string get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_sv,
                                  specfem::element::property_tag::anisotropic,
                                  false> &point_kernel,
    const type_real value) {
  return get_error_message(
      static_cast<specfem::point::kernels<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::property_tag::anisotropic, false> >(point_kernel),
      value);
}

// Template get_error_message specialization: elastic sh anisotropic (child)
template <>
std::string get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_sh,
                                  specfem::element::property_tag::anisotropic,
                                  false> &point_kernel,
    const type_real value) {
  return get_error_message(
      static_cast<specfem::point::kernels<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::property_tag::anisotropic, false> >(point_kernel),
      value);
}

// Template get_error_message specialization: acoustic isotropic
template <>
std::string get_error_message(
    const specfem::point::kernels<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, false> &point_kernel,
    const type_real value) {
  std::ostringstream message;

  message << "\n\t Expected: " << value;
  message << "\n\t Got: \n";
  message << "\t\trho = " << point_kernel.rho() << "\n";
  message << "\t\tkappa = " << point_kernel.kappa() << "\n";
  message << "\t\trhop = " << point_kernel.rhop() << "\n";
  message << "\t\talpha = " << point_kernel.alpha() << "\n";

  return message.str();
}

//
// ==================== HACKATHON TODO: UNCOMMENT TO ENABLE ==================
//

/* <--- REMOVE THIS LINE TO ENABLE THE CODE BELOW FOR ELECTROMAGNETIC


// Template get_error_message specialization: electromagnetic sv isotropic
template <>
std::string get_error_message(
    const specfem::point::kernels<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::electromagnetic_sv,
        specfem::element::property_tag::isotropic, false> &point_kernel,
    const type_real value) {
  std::ostringstream message;

  message << "\n\t Expected: " << value;
  message << "\n\t Got: \n";
  message << "\t\tm0 = " << point_kernel.m0() << "\n";
  message << "\t\te0 = " << point_kernel.e0() << "\n";
  message << "\t\te11 = " << point_kernel.e11() << "\n";
  message << "\t\te33 = " << point_kernel.e33() << "\n";
  message << "\t\tsig11 = " << point_kernel.sig11() << "\n";
  message << "\t\tsig33 = " << point_kernel.sig33() << "\n";
  message << "\t\tQe11 = " << point_kernel.Qe11() << "\n";
  message << "\t\tQe33 = " << point_kernel.Qe33() << "\n";
  message << "\t\tQs11 = " << point_kernel.Qs11() << "\n";
  message << "\t\tQs33 = " << point_kernel.Qs33() << "\n";

  return message.str();
}

*/// <--- REMOVE THIS LINE TO ENABLE THE CODE ABOVE FOR ELECTROMAGNETIC

/* <--- REMOVE THIS LINE TO ENABLE THE CODE BELOW FOR POROELASTIC

// Template get_error_message specialization: poroelastic isotropic
template <>
std::string get_error_message(
    const specfem::point::kernels<
        specfem::dimension::type::dim2,
specfem::element::medium_tag::poroelastic,
        specfem::element::property_tag::isotropic, false> &point_kernel,
    const type_real value) {
  std::ostringstream message;

  message << "\n\t Expected: " << value;
  message << "\n\t Got: \n";
  message << "\t\tphi = " << point_kernel.phi() << "\n";
  message << "\t\ttortuosity = " << point_kernel.tortuosity() << "\n";
  message << "\t\trho_s = " << point_kernel.rho_s() << "\n";
  message << "\t\trho_f = " << point_kernel.rho_f() << "\n";
  message << "\t\tkappa_s = " << point_kernel.kappa_s() << "\n";
  message << "\t\tkappa_f = " << point_kernel.kappa_f() << "\n";
  message << "\t\tkappa_fr = " << point_kernel.kappa_fr() << "\n";
  message << "\t\tmu_fr = " << point_kernel.mu_fr() << "\n";
  message << "\t\teta = " << point_kernel.eta() << "\n";
  message << "\t\tKxx = " << point_kernel.Kxx() << "\n";
  message << "\t\tKzz = " << point_kernel.Kzz() << "\n";
  message << "\t\tKxz = " << point_kernel.Kxz() << "\n";

  return message.str();
}

*/// <--- REMOVE THIS LINE TO ENABLE THE CODE ABOVE FOR POROELASTIC

//
// ==================== HACKATHON TODO END ===================================
//

// Template get_point_kernel (No SIMD -> No SIMD)
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::point::kernels<specfem::dimension::type::dim2, MediumTag, PropertyTag,
                        false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels);

// Template get_point_kernel (SIMD -> No SIMD)
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::point::kernels<specfem::dimension::type::dim2, MediumTag, PropertyTag,
                        false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<specfem::dimension::type::dim2, MediumTag,
                                  PropertyTag, true> &point_kernel);

// Template get_point_kernel specialization:
//  elastic p-sv isotropic (No SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic_sv,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels) {

  const auto elastic_isotropic =
      kernels.get_container<specfem::element::medium_tag::elastic_sv,
                            specfem::element::property_tag::isotropic>();

  const int ispec_l = kernels.h_property_index_mapping(ispec);

  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_sv,
                          specfem::element::property_tag::isotropic, false>
      point_kernel;

  point_kernel.rho() = elastic_isotropic.h_rho(ispec_l, iz, ix);
  point_kernel.mu() = elastic_isotropic.h_mu(ispec_l, iz, ix);
  point_kernel.kappa() = elastic_isotropic.h_kappa(ispec_l, iz, ix);
  point_kernel.rhop() = elastic_isotropic.h_rhop(ispec_l, iz, ix);
  point_kernel.alpha() = elastic_isotropic.h_alpha(ispec_l, iz, ix);
  point_kernel.beta() = elastic_isotropic.h_beta(ispec_l, iz, ix);

  return point_kernel;
}

// Template get_point_kernel specialization:
//  elastic sh isotropic (No SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic_sh,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels) {

  const auto elastic_isotropic =
      kernels.get_container<specfem::element::medium_tag::elastic_sh,
                            specfem::element::property_tag::isotropic>();

  const int ispec_l = kernels.h_property_index_mapping(ispec);

  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_sh,
                          specfem::element::property_tag::isotropic, false>
      point_kernel;

  point_kernel.rho() = elastic_isotropic.h_rho(ispec_l, iz, ix);
  point_kernel.mu() = elastic_isotropic.h_mu(ispec_l, iz, ix);
  point_kernel.kappa() = elastic_isotropic.h_kappa(ispec_l, iz, ix);
  point_kernel.rhop() = elastic_isotropic.h_rhop(ispec_l, iz, ix);
  point_kernel.alpha() = elastic_isotropic.h_alpha(ispec_l, iz, ix);
  point_kernel.beta() = elastic_isotropic.h_beta(ispec_l, iz, ix);

  return point_kernel;
}

// Template get_point_kernel specialization:
//   elastic p-sv isotropic (SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic_sv,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_sv,
                                  specfem::element::property_tag::isotropic,
                                  true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_sv,
                          specfem::element::property_tag::isotropic, false>
      point_kernel_l;

  point_kernel_l.rho() = point_kernel.rho()[lane];
  point_kernel_l.mu() = point_kernel.mu()[lane];
  point_kernel_l.kappa() = point_kernel.kappa()[lane];
  point_kernel_l.rhop() = point_kernel.rhop()[lane];
  point_kernel_l.alpha() = point_kernel.alpha()[lane];
  point_kernel_l.beta() = point_kernel.beta()[lane];

  return point_kernel_l;
}

// Template get_point_kernel specialization:
//   elastic sh isotropic (SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic_sh,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_sh,
                                  specfem::element::property_tag::isotropic,
                                  true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_sh,
                          specfem::element::property_tag::isotropic, false>
      point_kernel_l;

  point_kernel_l.rho() = point_kernel.rho()[lane];
  point_kernel_l.mu() = point_kernel.mu()[lane];
  point_kernel_l.kappa() = point_kernel.kappa()[lane];
  point_kernel_l.rhop() = point_kernel.rhop()[lane];
  point_kernel_l.alpha() = point_kernel.alpha()[lane];
  point_kernel_l.beta() = point_kernel.beta()[lane];

  return point_kernel_l;
}

// Template get_point_kernel specialization:
//   elastic p-sv anisotropic (No SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic_sv,
                        specfem::element::property_tag::anisotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels) {

  const auto elastic_anisotropic =
      kernels.get_container<specfem::element::medium_tag::elastic_sv,
                            specfem::element::property_tag::anisotropic>();

  const int ispec_l = kernels.h_property_index_mapping(ispec);

  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_sv,
                          specfem::element::property_tag::anisotropic, false>
      point_kernel;

  point_kernel.rho() = elastic_anisotropic.h_rho(ispec_l, iz, ix);
  point_kernel.c11() = elastic_anisotropic.h_c11(ispec_l, iz, ix);
  point_kernel.c13() = elastic_anisotropic.h_c13(ispec_l, iz, ix);
  point_kernel.c15() = elastic_anisotropic.h_c15(ispec_l, iz, ix);
  point_kernel.c33() = elastic_anisotropic.h_c33(ispec_l, iz, ix);
  point_kernel.c35() = elastic_anisotropic.h_c35(ispec_l, iz, ix);
  point_kernel.c55() = elastic_anisotropic.h_c55(ispec_l, iz, ix);

  return point_kernel;
}

// Template get_point_kernel specialization:
//   elastic sh anisotropic (No SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic_sh,
                        specfem::element::property_tag::anisotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels) {

  const auto elastic_anisotropic =
      kernels.get_container<specfem::element::medium_tag::elastic_sh,
                            specfem::element::property_tag::anisotropic>();

  const int ispec_l = kernels.h_property_index_mapping(ispec);

  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_sh,
                          specfem::element::property_tag::anisotropic, false>
      point_kernel;

  point_kernel.rho() = elastic_anisotropic.h_rho(ispec_l, iz, ix);
  point_kernel.c11() = elastic_anisotropic.h_c11(ispec_l, iz, ix);
  point_kernel.c13() = elastic_anisotropic.h_c13(ispec_l, iz, ix);
  point_kernel.c15() = elastic_anisotropic.h_c15(ispec_l, iz, ix);
  point_kernel.c33() = elastic_anisotropic.h_c33(ispec_l, iz, ix);
  point_kernel.c35() = elastic_anisotropic.h_c35(ispec_l, iz, ix);
  point_kernel.c55() = elastic_anisotropic.h_c55(ispec_l, iz, ix);

  return point_kernel;
}

// Template get_point_kernel specialization:
//   elastic p-sv anisotropic (SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic_sv,
                        specfem::element::property_tag::anisotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_sv,
                                  specfem::element::property_tag::anisotropic,
                                  true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_sv,
                          specfem::element::property_tag::anisotropic, false>
      point_kernel_l;

  point_kernel_l.rho() = point_kernel.rho()[lane];
  point_kernel_l.c11() = point_kernel.c11()[lane];
  point_kernel_l.c13() = point_kernel.c13()[lane];
  point_kernel_l.c15() = point_kernel.c15()[lane];
  point_kernel_l.c33() = point_kernel.c33()[lane];
  point_kernel_l.c35() = point_kernel.c35()[lane];
  point_kernel_l.c55() = point_kernel.c55()[lane];

  return point_kernel_l;
}

// Template get_point_kernel specialization:
//   elastic sh anisotropic (SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic_sh,
                        specfem::element::property_tag::anisotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_sh,
                                  specfem::element::property_tag::anisotropic,
                                  true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_sh,
                          specfem::element::property_tag::anisotropic, false>
      point_kernel_l;

  point_kernel_l.rho() = point_kernel.rho()[lane];
  point_kernel_l.c11() = point_kernel.c11()[lane];
  point_kernel_l.c13() = point_kernel.c13()[lane];
  point_kernel_l.c15() = point_kernel.c15()[lane];
  point_kernel_l.c33() = point_kernel.c33()[lane];
  point_kernel_l.c35() = point_kernel.c35()[lane];
  point_kernel_l.c55() = point_kernel.c55()[lane];

  return point_kernel_l;
}

// Template get_point_kernel specialization:
//   acoustic isotropic (No SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::acoustic,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels) {

  const auto acoustic_isotropic =
      kernels.get_container<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic>();

  const int ispec_l = kernels.h_property_index_mapping(ispec);

  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic, false>
      point_kernel;

  point_kernel.rho() = acoustic_isotropic.h_rho(ispec_l, iz, ix);
  point_kernel.kappa() = acoustic_isotropic.h_kappa(ispec_l, iz, ix);
  point_kernel.alpha() = acoustic_isotropic.h_alpha(ispec_l, iz, ix);
  point_kernel.rhop() = acoustic_isotropic.h_rhop(ispec_l, iz, ix);

  return point_kernel;
}

// Template get_point_kernel specialization:
//   acoustic isotropic (SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::acoustic,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic, false>
      point_kernel_l;

  point_kernel_l.rho() = point_kernel.rho()[lane];
  point_kernel_l.kappa() = point_kernel.kappa()[lane];
  point_kernel_l.alpha() = point_kernel.alpha()[lane];
  point_kernel_l.rhop() = point_kernel.rhop()[lane];

  return point_kernel_l;
}

//
// ==================== HACKATHON TODO: UNCOMMENT TO ENABLE ==================
//

/* <--- REMOVE THIS LINE TO ENABLE THE CODE BELOW FOR EM

// Template get_point_kernel specialization:
//   electromagnetic sv isotropic (No SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::electromagnetic_sv,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                  const specfem::compute::kernels &kernels) {

    const auto electromagnetic_isotropic =
        kernels.get_container<specfem::element::medium_tag::electromagnetic_sv,
                              specfem::element::property_tag::isotropic>();

    const int ispec_l = kernels.h_property_index_mapping(ispec);

    specfem::point::kernels<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::electromagnetic_sv,
                            specfem::element::property_tag::isotropic, false>
        point_kernel;

    point_kernel.m0() = electromagnetic_isotropic.h_m0(ispec_l, iz, ix);
    point_kernel.e0() = electromagnetic_isotropic.h_e0(ispec_l, iz, ix);
    point_kernel.e11() = electromagnetic_isotropic.h_e11(ispec_l, iz, ix);
    point_kernel.e33() = electromagnetic_isotropic.h_e33(ispec_l, iz, ix);
    point_kernel.sig11() = electromagnetic_isotropic.h_sig11(ispec_l, iz, ix);
    point_kernel.sig33() = electromagnetic_isotropic.h_sig33(ispec_l, iz, ix);
    point_kernel.Qe11() = electromagnetic_isotropic.h_Qe11(ispec_l, iz, ix);
    point_kernel.Qe33() = electromagnetic_isotropic.h_Qe33(ispec_l, iz, ix);
    point_kernel.Qs11() = electromagnetic_isotropic.h_Qs11(ispec_l, iz, ix);
    point_kernel.Qs33() = electromagnetic_isotropic.h_Qs33(ispec_l, iz, ix);

    return point_kernel;
  }

// Template get_point_kernel specialization:
//   electromagnetic sv isotropic (SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::electromagnetic_sv,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::electromagnetic_sv,
        specfem::element::property_tag::isotropic, true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::electromagnetic_sv,
                          specfem::element::property_tag::isotropic, false>
      point_kernel_l;

    point_kernel_l.m0() = point_kernel.m0()[lane];
    point_kernel_l.e0() = point_kernel.e0()[lane];
    point_kernel_l.e11() = point_kernel.e11()[lane];
    point_kernel_l.e33() = point_kernel.e33()[lane];
    point_kernel_l.sig11() = point_kernel.sig11()[lane];
    point_kernel_l.sig33() = point_kernel.sig33()[lane];
    point_kernel_l.Qe11() = point_kernel.Qe11()[lane];
    point_kernel_l.Qe33() = point_kernel.Qe33()[lane];
    point_kernel_l.Qs11() = point_kernel.Qs11()[lane];
    point_kernel_l.Qs33() = point_kernel.Qs33()[lane];

    return point_kernel_l;
  }


*/// <--- REMOVE THIS LINE TO ENABLE THE CODE ABOVE FOR EM

/* <--- REMOVE THIS LINE TO ENABLE THE CODE BELOW FOR PORO

// Template get_point_kernel specialization:
//   poroelastic isotropic (No SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::poroelastic,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                  const specfem::compute::kernels &kernels) {

    const auto poroelastic_isotropic =
        kernels.get_container<specfem::element::medium_tag::poroelastic,
                              specfem::element::property_tag::isotropic>();

    const int ispec_l = kernels.h_property_index_mapping(ispec);

    specfem::point::kernels<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::poroelastic,
                            specfem::element::property_tag::isotropic, false>
        point_kernel;

    point_kernel.phi() = poroelastic_isotropic.h_phi(ispec_l, iz, ix);
    point_kernel.tortuosity() = poroelastic_isotropic.h_tortuosity(ispec_l, iz,
ix); point_kernel.rho_s() = poroelastic_isotropic.h_rho_s(ispec_l, iz, ix);
    point_kernel.rho_f() = poroelastic_isotropic.h_rho_f(ispec_l, iz, ix);
    point_kernel.kappa_s() = poroelastic_isotropic.h_kappa_s(ispec_l, iz, ix);
    point_kernel.kappa_f() = poroelastic_isotropic.h_kappa_f(ispec_l, iz, ix);
    point_kernel.kappa_fr() = poroelastic_isotropic.h_kappa_fr(ispec_l, iz, ix);
    point_kernel.mu_fr() = poroelastic_isotropic.h_mu_fr(ispec_l, iz, ix);
    point_kernel.eta() = poroelastic_isotropic.h_eta(ispec_l, iz, ix);
    point_kernel.Kxx() = poroelastic_isotropic.h_Kxx(ispec_l, iz, ix);
    point_kernel.Kzz() = poroelastic_isotropic.h_Kzz(ispec_l, iz, ix);
    point_kernel.Kxz() = poroelastic_isotropic.h_Kxz(ispec_l, iz, ix);

    return point_kernel;
  }

// Template get_point_kernel specialization:
//   poroelastic isotropic (SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::poroelastic,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::poroelastic,
        specfem::element::property_tag::isotropic, true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::poroelastic,
                          specfem::element::property_tag::isotropic, false>
      point_kernel_l;

    point_kernel_l.phi() = point_kernel.phi()[lane];
    point_kernel_l.tortuosity() = point_kernel.tortuosity()[lane];
    point_kernel_l.rho_s() = point_kernel.rho_s()[lane];
    point_kernel_l.rho_f() = point_kernel.rho_f()[lane];
    point_kernel_l.kappa_s() = point_kernel.kappa_s()[lane];
    point_kernel_l.kappa_f() = point_kernel.kappa_f()[lane];
    point_kernel_l.kappa_fr() = point_kernel.kappa_fr()[lane];
    point_kernel_l.mu_fr() = point_kernel.mu_fr()[lane];
    point_kernel_l.eta() = point_kernel.eta()[lane];
    point_kernel_l.Kxx() = point_kernel.Kxx()[lane];
    point_kernel_l.Kzz() = point_kernel.Kzz()[lane];
    point_kernel_l.Kxz() = point_kernel.Kxz()[lane];

    return point_kernel_l;
  }

*/// <--- REMOVE THIS LINE TO ENABLE THE CODE ABOVE FOR PORO

//
// ==================== HACKATHON TODO END ===================================
//

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename IndexViewType, typename ValueViewType>
void check_to_value(const specfem::compute::element_types &element_types,
                    const specfem::compute::kernels kernels,
                    const IndexViewType &ispecs,
                    const ValueViewType &values_to_store) {
  const int nspec = kernels.nspec;
  const int ngllx = kernels.ngllx;
  const int ngllz = kernels.ngllz;

  const auto elements =
      element_types.get_elements_on_host(MediumTag, PropertyTag);

  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;
  constexpr int nprops = PointType::nprops;

  for (int i = 0; i < ispecs.extent(0); ++i) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        const int ielement = ispecs(i);
        const int n_simd_elements = (simd_size + ielement > elements.extent(0))
                                        ? elements.extent(0) - ielement
                                        : simd_size;
        for (int j = 0; j < n_simd_elements; j++) {
          const auto point_kernel = get_point_kernel<MediumTag, PropertyTag>(
              ielement + j, iz, ix, kernels);
          const type_real value = values_to_store(i);
          for (int l = 0; l < nprops; l++) {
            if (std::abs(point_kernel.data[l] - value) > 1e-6) {
              std::ostringstream message;

              message << "\n \t Error at ispec = " << ielement + j
                      << ", iz = " << iz << ", ix = " << ix;
              message << get_error_message(point_kernel, value);

              throw std::runtime_error(message.str());
            }
          }
        }
      }
    }
  }

  return;
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool Store, bool Add,
          bool using_simd, typename IndexViewType, typename ValueViewType>
void execute_store_or_add(specfem::compute::kernels &kernels,
                          const int element_size, const IndexViewType &ispecs,
                          const ValueViewType &values_to_store) {

  const int nspec = kernels.nspec;
  const int ngllx = kernels.ngllx;
  const int ngllz = kernels.ngllz;

  const int N = ispecs.extent(0);

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;

  Kokkos::parallel_for(
      "check_store_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3> >(
          { 0, 0, 0 }, { N, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        const int ielement = ispecs(i);
        constexpr int simd_size = PointType::simd::size();
        auto &kernels_l = kernels;
        const int n_simd_elements = (simd_size + ielement > element_size)
                                        ? element_size - ielement
                                        : simd_size;

        const auto index =
            get_index<using_simd>(ielement, n_simd_elements, iz, ix);
        const type_real value = values_to_store(i);
        PointType point;
        for (int l = 0; l < PointType::nprops; l++) {
          point.data[l] = value;
        }
        if constexpr (Store) {
          specfem::compute::store_on_device(index, point, kernels_l);
        } else if constexpr (Add) {
          specfem::compute::add_on_device(index, point, kernels_l);
        }
      });

  Kokkos::fence();
  kernels.copy_to_host();
  return;
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd>
void check_store_and_add(specfem::compute::kernels &kernels,
                         const specfem::compute::element_types &element_types) {

  const int nspec = kernels.nspec;
  const int ngllx = kernels.ngllx;
  const int ngllz = kernels.ngllz;

  const auto elements =
      element_types.get_elements_on_host(MediumTag, PropertyTag);

  // Evaluate at N evenly spaced points
  constexpr int N = 20;

  if (elements.extent(0) < N) {
    return;
  }

  Kokkos::View<int[N], Kokkos::DefaultExecutionSpace> ispecs("ispecs");
  Kokkos::View<type_real[N], Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store");
  auto ispecs_h = Kokkos::create_mirror_view(ispecs);
  auto values_to_store_h = Kokkos::create_mirror_view(values_to_store);

  const int element_size = elements.extent(0);
  const int step = element_size / N;

  for (int i = 0; i < N; i++) {
    ispecs_h(i) = elements(i * step);
    values_to_store_h(i) = 10.5 + i;
  }

  ispecs_h(N - 1) = elements(element_size - 5); // check when simd is not full

  Kokkos::deep_copy(ispecs, ispecs_h);
  Kokkos::deep_copy(values_to_store, values_to_store_h);

  execute_store_or_add<MediumTag, PropertyTag, true, false, using_simd>(
      kernels, element_size, ispecs, values_to_store);

  check_to_value<MediumTag, PropertyTag, using_simd>(
      element_types, kernels, ispecs_h, values_to_store_h);

  execute_store_or_add<MediumTag, PropertyTag, false, true, using_simd>(
      kernels, element_size, ispecs, values_to_store);

  for (int i = 0; i < N; i++) {
    values_to_store_h(i) *= 2;
  }

  check_to_value<MediumTag, PropertyTag, using_simd>(
      element_types, kernels, ispecs_h, values_to_store_h);
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd>
void check_load_on_device(
    specfem::compute::kernels &kernels,
    const specfem::compute::element_types &element_types) {
  const int nspec = kernels.nspec;
  const int ngllx = kernels.ngllx;
  const int ngllz = kernels.ngllz;

  const auto elements =
      element_types.get_elements_on_host(MediumTag, PropertyTag);

  // Evaluate at N evenly spaced points
  constexpr int N = 20;

  if (elements.extent(0) < N) {
    return;
  }

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;

  Kokkos::View<int[N], Kokkos::DefaultExecutionSpace> ispecs("ispecs");
  Kokkos::View<type_real[N], Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store");
  auto ispecs_h = Kokkos::create_mirror_view(ispecs);
  auto values_to_store_h = Kokkos::create_mirror_view(values_to_store);

  const int element_size = elements.extent(0);
  const int step = element_size / N;
  constexpr int nprops = PointType::nprops;

  for (int i = 0; i < N; i++) {
    ispecs_h(i) = elements(i * step);
    values_to_store_h(i) = 2 * (10.5 + i);
  }

  ispecs_h(N - 1) = elements(element_size - 5); // check when simd is not full

  Kokkos::deep_copy(ispecs, ispecs_h);

  Kokkos::View<PointType **[N], Kokkos::DefaultExecutionSpace> point_kernels(
      "point_kernels", ngllz, ngllx);
  auto h_point_kernels = Kokkos::create_mirror_view(point_kernels);

  Kokkos::parallel_for(
      "check_load_on_device",
      Kokkos::MDRangePolicy<Kokkos::Rank<3> >({ 0, 0, 0 }, { N, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        const int ielement = ispecs(i);
        constexpr int simd_size = PointType::simd::size();
        const int n_simd_elements = (simd_size + ielement > element_size)
                                        ? element_size - ielement
                                        : simd_size;

        const auto index =
            get_index<using_simd>(ielement, n_simd_elements, iz, ix);
        PointType point;
        specfem::compute::load_on_device(index, kernels, point);
        point_kernels(iz, ix, i) = point;
      });

  Kokkos::fence();
  Kokkos::deep_copy(h_point_kernels, point_kernels);

  for (int i = 0; i < N; i++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        using simd = specfem::datatype::simd<type_real, using_simd>;
        const auto &point_kernel = h_point_kernels(iz, ix, i);
        const int ielement = ispecs_h(i);
        constexpr int simd_size = PointType::simd::size();
        const int n_simd_elements = (simd_size + ielement > element_size)
                                        ? element_size - ielement
                                        : simd_size;
        const type_real value_l = values_to_store_h(i);
        if constexpr (using_simd) {
          for (int lane = 0; lane < n_simd_elements; lane++) {
            const auto point_kernel_l = get_point_kernel(lane, point_kernel);
            for (int l = 0; l < nprops; l++) {
              if (std::abs(point_kernel_l.data[l] - value_l) > 1e-6) {
                std::ostringstream message;

                message << "\n \t Error in function load_on_device";

                message << "\n \t Error at ispec = " << ielement
                        << ", iz = " << 0 << ", ix = " << 0;
                message << get_error_message(point_kernel_l, value_l);

                throw std::runtime_error(message.str());
              }
            }
          }
        } else if constexpr (!using_simd) {
          for (int l = 0; l < nprops; l++) {
            if (std::abs(point_kernel.data[l] - value_l) > 1e-6) {
              std::ostringstream message;

              message << "\n \t Error in function load_on_device";

              message << "\n \t Error at ispec = " << ielement << ", iz = " << 0
                      << ", ix = " << 0;
              message << get_error_message(point_kernel, value_l);

              throw std::runtime_error(message.str());
            }
          }
        }
      }
    }
  }

  return;
}

void test_kernels(specfem::compute::assembly &assembly) {

  const auto &element_types = assembly.element_types;
  auto &kernels = assembly.kernels;

  //
  // ==================== HACKATHON TODO: ADD MEDIUM_TAG_POROELASTIC ==========
  //

  //
  // ==================== HACKATHON TODO: ADD MEDIUM_TAG_ELECTROMAGNETIC_SV ===
  //

#define TEST_STORE_AND_ADD(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)            \
  check_store_and_add<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), false>(      \
      kernels, element_types);                                                 \
  check_load_on_device<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), false>(     \
      kernels, element_types);                                                 \
  check_store_and_add<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), true>(       \
      kernels, element_types);                                                 \
  check_load_on_device<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), true>(      \
      kernels, element_types);

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      TEST_STORE_AND_ADD,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef TEST_STORE_AND_ADD
}

TEST_F(ASSEMBLY, kernels_device_functions) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    specfem::compute::assembly assembly = std::get<5>(parameters);

    try {
      test_kernels(assembly);

      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m " << Test.name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
    } catch (std::exception &e) {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;31m[FAILED]\033[0m \n"
                << "-------------------------------------------------------\n"
                << "- Test: " << Test.name << "\n"
                << "- Error: " << e.what() << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      ADD_FAILURE();
    }
  }
}
