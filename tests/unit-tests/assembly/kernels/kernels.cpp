#include "point/kernels.hpp"
#include "../test_fixture/test_fixture.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "policies/chunk.hpp"
#include "specfem_setup.hpp"
#include <gtest/gtest.h>

// Primary template
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd = false>
std::string get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2, MediumTag,
                                  PropertyTag, false> &point_kernel,
    const type_real value);

// Specialization for elastic isotropic case
template <specfem::element::medium_tag MediumTag, bool using_simd = false>
std::enable_if_t<(MediumTag == specfem::element::medium_tag::elastic_psv ||
                  MediumTag == specfem::element::medium_tag::elastic_sh) &&
                     using_simd == false,
                 std::string>
get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2, MediumTag,
                                  specfem::element::property_tag::isotropic,
                                  false> &point_kernel,
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

// Specialization for elastic anisotropic case
template <specfem::element::medium_tag MediumTag, bool using_simd = false>
std::enable_if_t<(MediumTag == specfem::element::medium_tag::elastic_psv ||
                  MediumTag == specfem::element::medium_tag::elastic_sh) &&
                     using_simd == false,
                 std::string>
get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2, MediumTag,
                                  specfem::element::property_tag::anisotropic,
                                  false> &point_kernel,
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

/* <--- Remove this line when em kernels are implemented


// Template get_error_message specialization: electromagnetic te isotropic
template <>
std::string get_error_message(
    const specfem::point::kernels<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::electromagnetic_te,
        specfem::element::property_tag::isotropic, false> &point_kernel,
    const type_real value) {
  std::ostringstream message;

   //
  //  Here go the parameters for EM Kernels
  //

  std::ostringstream message;

  message << "EM Kernels are not implemented yet. Please implement them first.";
          << "[" << __FILE__ << ":" << __LINE__ << "]\n";
  std::throw std::runtime_error(message.str());

  message << "\n\t Expected: " << value;
  message << "\n\t Got: \n";

  return message.str();
}

*/// <--- REMOVE THIS LINE TO ENABLE THE CODE ABOVE FOR ELECTROMAGNETIC

// Template get_error_message specialization: poroelastic isotropic
template <>
std::string get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::poroelastic,
                                  specfem::element::property_tag::isotropic,
                                  false> &point_kernel,
    const type_real value) {
  std::ostringstream message;

  message << "\n\t Expected: " << value;
  message << "\n\t Got: \n";
  message << "\t\trhot = " << point_kernel.rhot() << "\n";
  message << "\t\trhof = " << point_kernel.rhof() << "\n";
  message << "\t\teta = " << point_kernel.eta() << "\n";
  message << "\t\tsm = " << point_kernel.sm() << "\n";
  message << "\t\tmu_fr = " << point_kernel.mu_fr() << "\n";
  message << "\t\tB = " << point_kernel.B() << "\n";
  message << "\t\tC = " << point_kernel.C() << "\n";
  message << "\t\tM = " << point_kernel.M() << "\n";
  message << "\t\tmu_frb = " << point_kernel.mu_frb() << "\n";
  message << "\t\trhob = " << point_kernel.rhob() << "\n";
  message << "\t\trhofb = " << point_kernel.rhofb() << "\n";
  message << "\t\tphi = " << point_kernel.phi() << "\n";
  message << "\t\tcpI = " << point_kernel.cpI() << "\n";
  message << "\t\tcpII = " << point_kernel.cpII() << "\n";
  message << "\t\tcs = " << point_kernel.cs() << "\n";
  message << "\t\trhobb = " << point_kernel.rhobb() << "\n";
  message << "\t\trhofbb = " << point_kernel.rhofbb() << "\n";
  message << "\t\tratio = " << point_kernel.ratio() << "\n";
  message << "\t\tphib = " << point_kernel.phib() << "\n";

  return message.str();
}

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
                        specfem::element::medium_tag::elastic_psv,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels) {

  const auto elastic_isotropic =
      kernels.get_container<specfem::element::medium_tag::elastic_psv,
                            specfem::element::property_tag::isotropic>();

  const int ispec_l = kernels.h_property_index_mapping(ispec);

  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_psv,
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
                        specfem::element::medium_tag::elastic_psv,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_psv,
                                  specfem::element::property_tag::isotropic,
                                  true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_psv,
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
                        specfem::element::medium_tag::elastic_psv,
                        specfem::element::property_tag::anisotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels) {

  const auto elastic_anisotropic =
      kernels.get_container<specfem::element::medium_tag::elastic_psv,
                            specfem::element::property_tag::anisotropic>();

  const int ispec_l = kernels.h_property_index_mapping(ispec);

  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_psv,
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
                        specfem::element::medium_tag::elastic_psv,
                        specfem::element::property_tag::anisotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::elastic_psv,
                                  specfem::element::property_tag::anisotropic,
                                  true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic_psv,
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
//   electromagnetic te isotropic (No SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::electromagnetic_te,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                  const specfem::compute::kernels &kernels) {

    const auto electromagnetic_isotropic =
        kernels.get_container<specfem::element::medium_tag::electromagnetic_te,
                              specfem::element::property_tag::isotropic>();

    const int ispec_l = kernels.h_property_index_mapping(ispec);

    specfem::point::kernels<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::electromagnetic_te,
                            specfem::element::property_tag::isotropic, false>
        point_kernel;

    //
    // Here go the parameters for EM Kernels
    //
    // point_kernel.<param>() = electromagnetic_isotropic.h_<param>(ispec_l, iz,
ix);

    kokkos::abort("EM Kernels not implemented.");

    return point_kernel;
  }

// Template get_point_kernel specialization:
//   electromagnetic te isotropic (SIMD -> No SIMD)
template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::electromagnetic_te,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::electromagnetic_te,
        specfem::element::property_tag::isotropic, true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::electromagnetic_te,
                          specfem::element::property_tag::isotropic, false>
      point_kernel_l;

    //
    // Here go the parameters for EM Kernels
    //
    // point_kernel_l.<param>() = point_kernel.<param>()[lane];

    kokkos::abort("EM Kernels not implemented for SIMD");

    return point_kernel_l;
  }


*/// <--- REMOVE THIS LINE TO ENABLE THE CODE ABOVE FOR EM

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

  point_kernel.rhot() = poroelastic_isotropic.h_rhot(ispec_l, iz, ix);
  point_kernel.rhof() = poroelastic_isotropic.h_rhof(ispec_l, iz, ix);
  point_kernel.eta() = poroelastic_isotropic.h_eta(ispec_l, iz, ix);
  point_kernel.sm() = poroelastic_isotropic.h_sm(ispec_l, iz, ix);
  point_kernel.mu_fr() = poroelastic_isotropic.h_mu_fr(ispec_l, iz, ix);
  point_kernel.B() = poroelastic_isotropic.h_B(ispec_l, iz, ix);
  point_kernel.C() = poroelastic_isotropic.h_C(ispec_l, iz, ix);
  point_kernel.M() = poroelastic_isotropic.h_M(ispec_l, iz, ix);
  point_kernel.mu_frb() = poroelastic_isotropic.h_mu_frb(ispec_l, iz, ix);
  point_kernel.rhob() = poroelastic_isotropic.h_rhob(ispec_l, iz, ix);
  point_kernel.rhofb() = poroelastic_isotropic.h_rhofb(ispec_l, iz, ix);
  point_kernel.phi() = poroelastic_isotropic.h_phi(ispec_l, iz, ix);
  point_kernel.cpI() = poroelastic_isotropic.h_cpI(ispec_l, iz, ix);
  point_kernel.cpII() = poroelastic_isotropic.h_cpII(ispec_l, iz, ix);
  point_kernel.cs() = poroelastic_isotropic.h_cs(ispec_l, iz, ix);
  point_kernel.rhobb() = poroelastic_isotropic.h_rhobb(ispec_l, iz, ix);
  point_kernel.rhofbb() = poroelastic_isotropic.h_rhofbb(ispec_l, iz, ix);
  point_kernel.ratio() = poroelastic_isotropic.h_ratio(ispec_l, iz, ix);
  point_kernel.phib() = poroelastic_isotropic.h_phib(ispec_l, iz, ix);

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
    const specfem::point::kernels<specfem::dimension::type::dim2,
                                  specfem::element::medium_tag::poroelastic,
                                  specfem::element::property_tag::isotropic,
                                  true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::poroelastic,
                          specfem::element::property_tag::isotropic, false>
      point_kernel_l;

  point_kernel_l.rhot() = point_kernel.rhot()[lane];
  point_kernel_l.rhof() = point_kernel.rhof()[lane];
  point_kernel_l.eta() = point_kernel.eta()[lane];
  point_kernel_l.sm() = point_kernel.sm()[lane];
  point_kernel_l.mu_fr() = point_kernel.mu_fr()[lane];
  point_kernel_l.B() = point_kernel.B()[lane];
  point_kernel_l.C() = point_kernel.C()[lane];
  point_kernel_l.M() = point_kernel.M()[lane];
  point_kernel_l.mu_frb() = point_kernel.mu_frb()[lane];
  point_kernel_l.rhob() = point_kernel.rhob()[lane];
  point_kernel_l.rhofb() = point_kernel.rhofb()[lane];
  point_kernel_l.phi() = point_kernel.phi()[lane];
  point_kernel_l.cpI() = point_kernel.cpI()[lane];
  point_kernel_l.cpII() = point_kernel.cpII()[lane];
  point_kernel_l.cs() = point_kernel.cs()[lane];
  point_kernel_l.rhobb() = point_kernel.rhobb()[lane];
  point_kernel_l.rhofbb() = point_kernel.rhofbb()[lane];
  point_kernel_l.ratio() = point_kernel.ratio()[lane];
  point_kernel_l.phib() = point_kernel.phib()[lane];

  return point_kernel_l;
}

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
  // ==================== HACKATHON TODO: ADD POROELASTIC ==========
  //

  //
  // == HACKATHON TODO: ADD ELECTROMAGNETIC_TE
  //                                           IFF EM KERNEL is implemented. ==
  //

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
      {
        check_store_and_add<_medium_tag_, _property_tag_, false>(kernels,
                                                                 element_types);
        check_load_on_device<_medium_tag_, _property_tag_, false>(
            kernels, element_types);
        check_store_and_add<_medium_tag_, _property_tag_, true>(kernels,
                                                                element_types);
        check_load_on_device<_medium_tag_, _property_tag_, true>(kernels,
                                                                 element_types);
      })
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
