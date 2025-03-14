#include "../test_fixture/test_fixture.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/property/reader.hpp"
#include "IO/property/writer.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "policies/chunk.hpp"
#include "specfem_setup.hpp"
#include <gtest/gtest.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#ifndef TEST_OUTPUT_DIR
#define TEST_OUTPUT_DIR "." // Fallback in case it's not set
#endif

inline void error_message_header(std::ostringstream &message,
                                 const type_real &value, const int &mode) {
  if (mode == 0) {
    message << "\n\t Expected: " << value;
    message << "\n\t Got: \n";
  } else if (mode == 1) {
    message << "\n\t Expected: ";
  } else if (mode == 2) {
    message << "\n\t Got: ";
  }
}

// Error message template to print properties
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd = false>
std::string get_error_message(
    const specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                     PropertyTag, false> &point_property,
    const type_real value, const int mode = 0);

// Template specialization: elastic isotropic (parent)
template <>
std::string get_error_message(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, false> &point_property,
    const type_real value, const int mode) {
  std::ostringstream message;

  error_message_header(message, value, mode);
  message << "\t\trho = " << point_property.rho() << "\n";
  message << "\t\tmu = " << point_property.mu() << "\n";
  message << "\t\tlambdaplus2mu = " << point_property.lambdaplus2mu() << "\n";

  return message.str();
}

// Template specialization: elastic anisotropic (parent)
template <>
std::string get_error_message(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, false> &point_property,
    const type_real value, const int mode) {
  std::ostringstream message;

  error_message_header(message, value, mode);
  message << "\t\trho = " << point_property.rho() << "\n";
  message << "\t\tc11 = " << point_property.c11() << "\n";
  message << "\t\tc13 = " << point_property.c13() << "\n";
  message << "\t\tc15 = " << point_property.c15() << "\n";
  message << "\t\tc33 = " << point_property.c33() << "\n";
  message << "\t\tc35 = " << point_property.c35() << "\n";
  message << "\t\tc55 = " << point_property.c55() << "\n";

  return message.str();
}

// Template specialization: elastic p_sv isotropic  (child)
template <>
std::string get_error_message(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sv,
                                     specfem::element::property_tag::isotropic,
                                     false> &point_property,
    const type_real value, const int mode) {

  return get_error_message(
      static_cast<specfem::point::properties<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::property_tag::isotropic, false> >(point_property),
      value, mode);
}

// Template specialization: elastic p_sv anisotropic (child)
template <>
std::string get_error_message(
    const specfem::point::properties<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sv,
        specfem::element::property_tag::anisotropic, false> &point_property,
    const type_real value, const int mode) {

  return get_error_message(
      static_cast<specfem::point::properties<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::property_tag::anisotropic, false> >(point_property),
      value, mode);
}

// Template specialization: elastic sh isotropic (child)
template <>
std::string get_error_message(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sh,
                                     specfem::element::property_tag::isotropic,
                                     false> &point_property,
    const type_real value, const int mode) {

  return get_error_message(
      static_cast<specfem::point::properties<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::property_tag::isotropic, false> >(point_property),
      value, mode);
}

// Template specialization: elastic sh anisotropic (child)
template <>
std::string get_error_message(
    const specfem::point::properties<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sh,
        specfem::element::property_tag::anisotropic, false> &point_property,
    const type_real value, const int mode) {

  return get_error_message(
      static_cast<specfem::point::properties<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::property_tag::anisotropic, false> >(point_property),
      value, mode);
}

// Template specialization: acoustic isotropic
template <>
std::string get_error_message(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, false> &point_property,
    const type_real value, const int mode) {
  std::ostringstream message;

  error_message_header(message, value, mode);
  message << "\t\trho_inverse = " << point_property.rho_inverse() << "\n";
  message << "\t\tkappa = " << point_property.kappa() << "\n";

  return message.str();
}

// Template get_point_property: No SIMD -> No SIMD
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                           PropertyTag, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties);

// Template get_point_property: SIMD -> No SIMD
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                           PropertyTag, false>
get_point_property(
    const int lane,
    const specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                     PropertyTag, true> &point_property);

// Template get_point_property specialization:
//    elastic p-sv isotropic (No SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sv,
                           specfem::element::property_tag::isotropic, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties) {

  const auto elastic_isotropic =
      properties.get_container<specfem::element::medium_tag::elastic_sv,
                               specfem::element::property_tag::isotropic>();

  const int ispec_l = properties.h_property_index_mapping(ispec);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic_sv,
                             specfem::element::property_tag::isotropic, false>
      point_property;

  point_property.rho() = elastic_isotropic.h_rho(ispec_l, iz, ix);
  point_property.mu() = elastic_isotropic.h_mu(ispec_l, iz, ix);
  point_property.lambdaplus2mu() =
      elastic_isotropic.h_lambdaplus2mu(ispec_l, iz, ix);

  return point_property;
}

// Template get_point_property specialization:
//    elastic sh isotropic (No SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sh,
                           specfem::element::property_tag::isotropic, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties) {

  const auto elastic_isotropic =
      properties.get_container<specfem::element::medium_tag::elastic_sh,
                               specfem::element::property_tag::isotropic>();

  const int ispec_l = properties.h_property_index_mapping(ispec);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic_sh,
                             specfem::element::property_tag::isotropic, false>
      point_property;

  point_property.rho() = elastic_isotropic.h_rho(ispec_l, iz, ix);
  point_property.mu() = elastic_isotropic.h_mu(ispec_l, iz, ix);
  point_property.lambdaplus2mu() =
      elastic_isotropic.h_lambdaplus2mu(ispec_l, iz, ix);

  return point_property;
}

// Template specialization:
//    elastic p-sv isotropic (SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sv,
                           specfem::element::property_tag::isotropic, false>
get_point_property(
    const int lane,
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sv,
                                     specfem::element::property_tag::isotropic,
                                     true> &point_property) {
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic_sv,
                             specfem::element::property_tag::isotropic, false>
      point_property_l;

  point_property_l.rho() = point_property.rho()[lane];
  point_property_l.mu() = point_property.mu()[lane];
  point_property_l.lambdaplus2mu() = point_property.lambdaplus2mu()[lane];

  return point_property_l;
}

// Template get_point_property specialization:
//    elastic sh isotropic (SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sh,
                           specfem::element::property_tag::isotropic, false>
get_point_property(
    const int lane,
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sh,
                                     specfem::element::property_tag::isotropic,
                                     true> &point_property) {
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic_sh,
                             specfem::element::property_tag::isotropic, false>
      point_property_l;

  point_property_l.rho() = point_property.rho()[lane];
  point_property_l.mu() = point_property.mu()[lane];
  point_property_l.lambdaplus2mu() = point_property.lambdaplus2mu()[lane];

  return point_property_l;
}

// Template get_point_property specialization:
//    elastic p-sv anisotropic (No SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sv,
                           specfem::element::property_tag::anisotropic, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties) {

  const auto elastic_anisotropic =
      properties.get_container<specfem::element::medium_tag::elastic_sv,
                               specfem::element::property_tag::anisotropic>();

  const int ispec_l = properties.h_property_index_mapping(ispec);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic_sv,
                             specfem::element::property_tag::anisotropic, false>
      point_property;

  point_property.rho() = elastic_anisotropic.h_rho(ispec_l, iz, ix);
  point_property.c11() = elastic_anisotropic.h_c11(ispec_l, iz, ix);
  point_property.c13() = elastic_anisotropic.h_c13(ispec_l, iz, ix);
  point_property.c15() = elastic_anisotropic.h_c15(ispec_l, iz, ix);
  point_property.c33() = elastic_anisotropic.h_c33(ispec_l, iz, ix);
  point_property.c35() = elastic_anisotropic.h_c35(ispec_l, iz, ix);
  point_property.c55() = elastic_anisotropic.h_c55(ispec_l, iz, ix);
  point_property.c12() = elastic_anisotropic.h_c12(ispec_l, iz, ix);
  point_property.c23() = elastic_anisotropic.h_c23(ispec_l, iz, ix);
  point_property.c25() = elastic_anisotropic.h_c25(ispec_l, iz, ix);

  return point_property;
}

// Template get_point_property specialization:
//   elastic sh anisotropic (No SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sh,
                           specfem::element::property_tag::anisotropic, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties) {

  const auto elastic_anisotropic =
      properties.get_container<specfem::element::medium_tag::elastic_sh,
                               specfem::element::property_tag::anisotropic>();

  const int ispec_l = properties.h_property_index_mapping(ispec);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic_sh,
                             specfem::element::property_tag::anisotropic, false>
      point_property;

  point_property.rho() = elastic_anisotropic.h_rho(ispec_l, iz, ix);
  point_property.c11() = elastic_anisotropic.h_c11(ispec_l, iz, ix);
  point_property.c13() = elastic_anisotropic.h_c13(ispec_l, iz, ix);
  point_property.c15() = elastic_anisotropic.h_c15(ispec_l, iz, ix);
  point_property.c33() = elastic_anisotropic.h_c33(ispec_l, iz, ix);
  point_property.c35() = elastic_anisotropic.h_c35(ispec_l, iz, ix);
  point_property.c55() = elastic_anisotropic.h_c55(ispec_l, iz, ix);
  point_property.c12() = elastic_anisotropic.h_c12(ispec_l, iz, ix);
  point_property.c23() = elastic_anisotropic.h_c23(ispec_l, iz, ix);
  point_property.c25() = elastic_anisotropic.h_c25(ispec_l, iz, ix);

  return point_property;
}

// Template get_point_property specialization:
//    elastic p-sv anisotropic (SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sv,
                           specfem::element::property_tag::anisotropic, false>
get_point_property(
    const int lane,
    const specfem::point::properties<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sv,
        specfem::element::property_tag::anisotropic, true> &point_property) {
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic_sv,
                             specfem::element::property_tag::anisotropic, false>
      point_property_l;

  point_property_l.rho() = point_property.rho()[lane];
  point_property_l.c11() = point_property.c11()[lane];
  point_property_l.c13() = point_property.c13()[lane];
  point_property_l.c15() = point_property.c15()[lane];
  point_property_l.c33() = point_property.c33()[lane];
  point_property_l.c35() = point_property.c35()[lane];
  point_property_l.c55() = point_property.c55()[lane];
  point_property_l.c12() = point_property.c12()[lane];
  point_property_l.c23() = point_property.c23()[lane];
  point_property_l.c25() = point_property.c25()[lane];

  return point_property_l;
}

// Template get_point_property specialization:
//    elastic sh anisotropic (SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sh,
                           specfem::element::property_tag::anisotropic, false>
get_point_property(
    const int lane,
    const specfem::point::properties<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sh,
        specfem::element::property_tag::anisotropic, true> &point_property) {
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic_sh,
                             specfem::element::property_tag::anisotropic, false>
      point_property_l;

  point_property_l.rho() = point_property.rho()[lane];
  point_property_l.c11() = point_property.c11()[lane];
  point_property_l.c13() = point_property.c13()[lane];
  point_property_l.c15() = point_property.c15()[lane];
  point_property_l.c33() = point_property.c33()[lane];
  point_property_l.c35() = point_property.c35()[lane];
  point_property_l.c55() = point_property.c55()[lane];
  point_property_l.c12() = point_property.c12()[lane];
  point_property_l.c23() = point_property.c23()[lane];
  point_property_l.c25() = point_property.c25()[lane];

  return point_property_l;
}

// Template get_point_property specialization:
//    acoustic isotropic (No SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::acoustic,
                           specfem::element::property_tag::isotropic, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties) {

  const auto acoustic_isotropic =
      properties.get_container<specfem::element::medium_tag::acoustic,
                               specfem::element::property_tag::isotropic>();

  const int ispec_l = properties.h_property_index_mapping(ispec);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::acoustic,
                             specfem::element::property_tag::isotropic, false>
      point_property;

  point_property.rho_inverse() =
      acoustic_isotropic.h_rho_inverse(ispec_l, iz, ix);
  point_property.kappa() = acoustic_isotropic.h_kappa(ispec_l, iz, ix);

  return point_property;
}

// Template get_point_property specialization:
//   acoustic isotropic (SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::acoustic,
                           specfem::element::property_tag::isotropic, false>
get_point_property(
    const int lane,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, true> &point_property) {
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::acoustic,
                             specfem::element::property_tag::isotropic, false>
      point_property_l;

  point_property_l.rho_inverse() = point_property.rho_inverse()[lane];
  point_property_l.kappa() = point_property.kappa()[lane];

  return point_property_l;
}

//
// ==================== HACKATHON TODO: UNCOMMENT TO ENABLE ==================
//

/* <--- REMOVE THIS LINE TO ENABLE THE CODE BELOW

// Template get_point_property specialization:
//    electromagnetic p-sv isotropic (No SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::electromagnetic_sv,
                           specfem::element::property_tag::isotropic, false>
get_point_property(
  const int ispec, const int iz, const int ix,
  const specfem::compute::properties &properties) {

  const auto electromagnetic_sv_isotropic =
      properties.get_container<specfem::element::medium_tag::electromagnetic_sv,
                               specfem::element::property_tag::isotropic>();

  const int ispec_l = properties.h_property_index_mapping(ispec);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::electromagnetic_sv,
                             specfem::element::property_tag::isotropic, false>
      point_property;

  point_property.mu0() = electromagnetic_sv_isotropic.h_mu0(ispec_l, iz, ix);
  point_property.e0() = electromagnetic_sv_isotropic.h_e0(ispec_l, iz, ix);
  point_property.e11() = electromagnetic_sv_isotropic.h_e11(ispec_l, iz, ix);
  point_property.e33() = electromagnetic_sv_isotropic.h_e33(ispec_l, iz, ix);
  point_property.s11() = electromagnetic_sv_isotropic.h_s11(ispec_l, iz, ix);
  point_property.s33() = electromagnetic_sv_isotropic.h_s33(ispec_l, iz, ix);
  point_porperty.Qe11() = electromagnetic_sv_isotropic.h_Qe11(ispec_l, iz, ix);
  point_porperty.Qe33() = electromagnetic_sv_isotropic.h_Qe33(ispec_l, iz, ix);
  point_porperty.Qs11() = electromagnetic_sv_isotropic.h_Qs11(ispec_l, iz, ix);
  point_porperty.Qs33() = electromagnetic_sv_isotropic.h_Qs33(ispec_l, iz, ix);

  return point_property;
}

// Template get_point_property specialization:
//    electromagnetic p-sv isotropic (SIMD -> No SIMD)
template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::electromagnetic_sv,
                           specfem::element::property_tag::isotropic, false>
get_point_property(
    const int lane,
    const specfem::point::properties<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::electromagnetic_sv,
        specfem::element::property_tag::isotropic, true> &point_property) {
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::electromagnetic_sv,
                             specfem::element::property_tag::isotropic, false>
      point_property_l;

  point_property_l.mu0() = point_property.mu0()[lane];
  point_property_l.e0() = point_property.e0()[lane];
  point_property_l.e11() = point_property.e11()[lane];
  point_property_l.e33() = point_property.e33()[lane];
  point_property_l.s11() = point_property.s11()[lane];
  point_property_l.s33() = point_property.s33()[lane];
  point_property_l.Qe11() = point_property.Qe11()[lane];
  point_property_l.Qe33() = point_property.Qe33()[lane];
  point_property_l.Qs11() = point_property.Qs11()[lane];
  point_property_l.Qs33() = point_property.Qs33()[lane];

  return point_property_l;
}

*/// <--- REMOVE THIS LINE TO ENABLE THE CODE ABOVE

//
// ==================== HACKATHON TODO END ===================================
//

template <bool using_simd>
void check_eq(
    const typename specfem::datatype::simd<type_real, using_simd>::datatype &p1,
    const typename specfem::datatype::simd<type_real, using_simd>::datatype &p2,
    const int &n_simd_elements, const std::string &msg) {
  if constexpr (using_simd) {
    for (int i = 0; i < n_simd_elements; i++) {
      if (p1[i] != p2[i]) {
        std::ostringstream message;

        message << "\n \t Error in function load_on_host (SIMD) | " << msg;
        message << "\n\t Expected: " << p1[i];
        message << "\n\t Got: " << p2[i];

        throw std::runtime_error(message.str());
      }
    }
  } else {
    if (p1 != p2) {
      std::ostringstream message;

      message << "\n \t Error in function load_on_host | " << msg;
      message << "\n\t Expected: " << p1;
      message << "\n\t Got: " << p2;

      throw std::runtime_error(message.str());
    }
  }
}

// Template check_point_properties
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd>
void check_point_properties(
    const specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                     PropertyTag, using_simd> &p1,
    const specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                     PropertyTag, using_simd> &p2,
    const int &n_simd_elements);

// Template check_point_properties specialization: elastic isotropic
template <bool using_simd>
void check_point_properties(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, using_simd> &p1,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, using_simd> &p2,
    const int &n_simd_elements) {
  check_eq<using_simd>(p1.rho(), p2.rho(), n_simd_elements, "rho");
  check_eq<using_simd>(p1.mu(), p2.mu(), n_simd_elements, ".mu");
  check_eq<using_simd>(p1.lambdaplus2mu(), p2.lambdaplus2mu(), n_simd_elements,
                       "lambdaplus2mu");
  check_eq<using_simd>(p1.lambda(),
                       p2.lambdaplus2mu() -
                           (static_cast<typename specfem::datatype::simd<
                                type_real, using_simd>::datatype>(2.0)) *
                               p2.mu(),
                       n_simd_elements, "lambda");
  check_eq<using_simd>(p1.rho_vp(), Kokkos::sqrt(p2.rho() * p2.lambdaplus2mu()),
                       n_simd_elements, "rho_vp");
  check_eq<using_simd>(p1.rho_vs(), Kokkos::sqrt(p2.rho() * p2.mu()),
                       n_simd_elements, "rho_vp");
}

// Template check_point_properties specialization: elastic anisotropic
template <bool using_simd>
void check_point_properties(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, using_simd> &p1,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, using_simd> &p2,
    const int &n_simd_elements) {
  check_eq<using_simd>(p1.rho(), p2.rho(), n_simd_elements, "rho");
  check_eq<using_simd>(p1.c11(), p2.c11(), n_simd_elements, "c11");
  check_eq<using_simd>(p1.c13(), p2.c13(), n_simd_elements, "c13");
  check_eq<using_simd>(p1.c15(), p2.c15(), n_simd_elements, "c15");
  check_eq<using_simd>(p1.c33(), p2.c33(), n_simd_elements, "c33");
  check_eq<using_simd>(p1.c35(), p2.c35(), n_simd_elements, "c35");
  check_eq<using_simd>(p1.c55(), p2.c55(), n_simd_elements, "c55");
  check_eq<using_simd>(p1.rho_vp(), Kokkos::sqrt(p2.rho() * p2.c33()),
                       n_simd_elements, "rho_vp");
  check_eq<using_simd>(p1.rho_vs(), Kokkos::sqrt(p2.rho() * p2.c55()),
                       n_simd_elements, "rho_vs");
}

// Template check_point_properties specialization: acoustic isotropic
template <bool using_simd>
void check_point_properties(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, using_simd> &p1,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, using_simd> &p2,
    const int &n_simd_elements) {

  check_eq<using_simd>(p1.rho_inverse(), p2.rho_inverse(), n_simd_elements,
                       "rho_inverse");
  check_eq<using_simd>(p1.kappa(), p2.kappa(), n_simd_elements, "kappa");
  check_eq<using_simd>(
      p1.kappa_inverse(),
      (static_cast<
          typename specfem::datatype::simd<type_real, using_simd>::datatype>(
          1.0)) /
          p2.kappa(),
      n_simd_elements, "kappa_inverse");
  check_eq<using_simd>(
      p1.rho_vpinverse(),
      Kokkos::sqrt(p2.rho_inverse() *
                   (static_cast<typename specfem::datatype::simd<
                        type_real, using_simd>::datatype>(1.0) /
                    p2.kappa())),
      n_simd_elements, "rho_vpinverse");
}

//
// ==================== HACKATHON TODO: UNCOMMENT TO ENABLE ==================
//

/* <--- REMOVE THIS LINE TO ENABLE THE CODE BELOW FOR EM

// Template check_point_properties specialization: electromagnetic sv isotropic
template <bool using_simd>
void check_point_properties(
    const specfem::point::properties<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::electromagnetic_sv,
        specfem::element::property_tag::isotropic, using_simd> &p1,
    const specfem::point::properties<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::electromagnetic_sv,
        specfem::element::property_tag::isotropic, using_simd> &p2,
    const int &n_simd_elements) {
  check_eq<using_simd>(p1.mu0(), p2.mu0(), n_simd_elements, "mu0");
  check_eq<using_simd>(p1.e0(), p2.e0(), n_simd_elements, "e0");
  check_eq<using_simd>(p1.e11(), p2.e11(), n_simd_elements, "e11");
  check_eq<using_simd>(p1.e33(), p2.e33(), n_simd_elements, "e33");
  check_eq<using_simd>(p1.s11(), p2.s11(), n_simd_elements, "s11");
  check_eq<using_simd>(p1.s33(), p2.s33(), n_simd_elements, "s33");
  check_eq<using_simd>(p1.Qe11(), p2.Qe11(), n_simd_elements, "Qe11");
  check_eq<using_simd>(p1.Qe33(), p2.Qe33(), n_simd_elements, "Qe33");
  check_eq<using_simd>(p1.Qs11(), p2.Qs11(), n_simd_elements, "Qs11");
  check_eq<using_simd>(p1.Qs33(), p2.Qs33(), n_simd_elements, "Qs33");
}


*/// <--- REMOVE THIS LINE TO ENABLE THE CODE ABOVE FOR EM

//
// ==================== HACKATHON TODO END ===================================
//

// Templatae check_to_value
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename IndexViewType, typename ValueViewType>
void check_to_value(const specfem::compute::properties properties,
                    specfem::compute::element_types element_types,
                    const IndexViewType &ispecs,
                    const ValueViewType &values_to_store) {
  const int nspec = properties.nspec;
  const int ngllx = properties.ngllx;
  const int ngllz = properties.ngllz;

  std::vector<int> elements;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if ((element_types.get_medium_tag(ispec) == MediumTag) &&
        (element_types.get_property_tag(ispec) == PropertyTag)) {
      elements.push_back(ispec);
    }
  }

  using PointType =
      specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                 PropertyTag, using_simd>;

  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  for (int i = 0; i < ispecs.extent(0); ++i) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        const int ielement = ispecs(i);
        const int n_simd_elements = (simd_size + ielement > elements.size())
                                        ? elements.size() - ielement
                                        : simd_size;
        for (int j = 0; j < n_simd_elements; j++) {
          const auto point_property =
              get_point_property<MediumTag, PropertyTag>(ielement + j, iz, ix,
                                                         properties);
          const type_real value = values_to_store(i);
          for (int l = 0; l < PointType::nprops; l++) {
            if (point_property.data[l] != value) {
              std::ostringstream message;

              message << "\n \t Error in function check_to_value";

              message << "\n \t Error at ispec = " << ielement + j
                      << ", iz = " << iz << ", ix = " << ix
                      << ", iprop = " << l;
              message << get_error_message(point_property, value);

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
          specfem::element::property_tag PropertyTag>
void check_compute_to_mesh(
    specfem::compute::assembly &assembly,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh) {
  const auto &properties = assembly.properties;
  const auto &mapping = assembly.mesh.mapping;
  const auto &materials = mesh.materials;
  const auto &element_types = assembly.element_types;

  const int nspec = properties.nspec;
  const int ngllx = properties.ngllx;
  const int ngllz = properties.ngllz;
  std::vector<int> elements;

  using PointType = specfem::point::properties<specfem::dimension::type::dim2,
                                               MediumTag, PropertyTag, false>;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if ((element_types.get_medium_tag(ispec) == MediumTag) &&
        (element_types.get_property_tag(ispec) == PropertyTag)) {
      elements.push_back(ispec);
    }
  }

  // Evaluate at N evenly spaced points
  constexpr int N = 20;

  if (elements.size() < N) {
    return;
  }

  const int element_size = elements.size();
  const int step = element_size / N;

  Kokkos::View<int[N], Kokkos::HostSpace> ispecs_h("ispecs_h", N);

  for (int i = 0; i < N; i++) {
    ispecs_h(i) = elements[i * step];
  }

  for (int i = 0; i < N; i++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        const int ielement = ispecs_h(i);
        const auto point_property = get_point_property<MediumTag, PropertyTag>(
            ielement, iz, ix, properties);
        const int ispec_mesh = mapping.compute_to_mesh(ielement);
        auto material =
            std::get<specfem::medium::material<MediumTag, PropertyTag> >(
                materials[ispec_mesh]);
        auto value = material.get_properties();
        for (int l = 0; l < PointType::nprops; l++) {
          if (point_property.data[l] != value.data[l]) {
            std::ostringstream message;

            message << "\n \t Error in function check_compute_to_mesh";

            message << "\n \t Error at ispec = " << ielement << ", iz = " << iz
                    << ", ix = " << ix << ", iprop = " << l;
            message << get_error_message(value, 0.0, 1);
            message << get_error_message(point_property, 0.0, 2);

            throw std::runtime_error(message.str());
          }
        }
        // if (point_property != value) {
        //   std::ostringstream message;

        //   message << "\n \t Error at ispec = " << ielement << ", iz = " << iz
        //           << ", ix = " << ix;

        //   message << get_error_message(value, 0.0, 1);
        //   message << get_error_message(point_property, 0.0, 2);

        //   throw std::runtime_error(message.str());
        // }
      }
    }
  }
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd>
void check_store_on_host(specfem::compute::properties &properties,
                         specfem::compute::element_types &element_types) {

  const int nspec = properties.nspec;
  const int ngllx = properties.ngllx;
  const int ngllz = properties.ngllz;
  std::vector<int> elements;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if ((element_types.get_medium_tag(ispec) == MediumTag) &&
        (element_types.get_property_tag(ispec) == PropertyTag)) {
      elements.push_back(ispec);
    }
  }

  // Evaluate at N evenly spaced points
  constexpr int N = 20;

  if (elements.size() < N) {
    return;
  }

  Kokkos::View<int[N], Kokkos::HostSpace> ispecs_h("ispecs_h", N);
  Kokkos::View<type_real[N], Kokkos::HostSpace> values_to_store_h(
      "values_to_store");

  const int element_size = elements.size();
  const int step = element_size / N;

  for (int i = 0; i < N; i++) {
    ispecs_h(i) = elements[i * step];
    values_to_store_h(i) = 10.5 + i;
  }

  ispecs_h(N - 1) = elements[element_size - 5]; // check when simd is not full

  using PointType =
      specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                 PropertyTag, using_simd>;

  for (int i = 0; i < N; i++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        const int ielement = ispecs_h(i);
        constexpr int simd_size = PointType::simd::size();
        const int n_simd_elements = (simd_size + ielement > element_size)
                                        ? element_size - ielement
                                        : simd_size;

        const auto index =
            get_index<using_simd>(ielement, n_simd_elements, iz, ix);
        const type_real value = values_to_store_h(i);
        PointType point;
        for (int l = 0; l < PointType::nprops; l++) {
          point.data[l] = value;
        }
        PointType point_loaded;
        specfem::compute::store_on_host(index, properties, point);
        specfem::compute::load_on_host(index, properties, point_loaded);
        check_point_properties<using_simd>(point_loaded, point,
                                           n_simd_elements);
      }
    }
  }

  check_to_value<MediumTag, PropertyTag, using_simd>(
      properties, element_types, ispecs_h, values_to_store_h);
  properties.copy_to_device();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd>
void check_load_on_device(specfem::compute::properties &properties,
                          specfem::compute::element_types &element_types) {
  const int nspec = properties.nspec;
  const int ngllx = properties.ngllx;
  const int ngllz = properties.ngllz;
  std::vector<int> elements;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if ((element_types.get_medium_tag(ispec) == MediumTag) &&
        (element_types.get_property_tag(ispec) == PropertyTag)) {
      elements.push_back(ispec);
    }
  }

  // Evaluate at N evenly spaced points
  constexpr int N = 20;

  if (elements.size() < N) {
    return;
  }

  using PointType =
      specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                 PropertyTag, using_simd>;

  Kokkos::View<int[N], Kokkos::DefaultExecutionSpace> ispecs("ispecs");
  Kokkos::View<type_real[N], Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store");
  auto ispecs_h = Kokkos::create_mirror_view(ispecs);
  auto values_to_store_h = Kokkos::create_mirror_view(values_to_store);

  const int element_size = elements.size();
  const int step = element_size / N;

  for (int i = 0; i < N; i++) {
    ispecs_h(i) = elements[i * step];
    values_to_store_h(i) = 10.5 + i;
  }

  ispecs_h(N - 1) = elements[element_size - 5]; // check when simd is not full

  Kokkos::deep_copy(ispecs, ispecs_h);

  Kokkos::View<PointType **[N], Kokkos::DefaultExecutionSpace> point_properties(
      "point_properties", ngllz, ngllx);
  auto h_point_properties = Kokkos::create_mirror_view(point_properties);

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
        specfem::compute::load_on_device(index, properties, point);
        point_properties(iz, ix, i) = point;
      });

  Kokkos::fence();
  Kokkos::deep_copy(h_point_properties, point_properties);

  for (int i = 0; i < N; i++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        using simd = specfem::datatype::simd<type_real, using_simd>;
        const auto &point_property = h_point_properties(iz, ix, i);
        const int ielement = ispecs_h(i);
        constexpr int simd_size = PointType::simd::size();
        const int n_simd_elements = (simd_size + ielement > element_size)
                                        ? element_size - ielement
                                        : simd_size;
        const type_real value_l = values_to_store_h(i);
        if constexpr (using_simd) {
          for (int lane = 0; lane < n_simd_elements; lane++) {
            const auto point_property_l =
                get_point_property(lane, point_property);
            for (int l = 0; l < PointType::nprops; l++) {
              if (std::abs(point_property_l.data[l] - value_l) > 1e-6) {
                std::ostringstream message;

                message << "\n \t Error in function load_on_device";

                message << "\n \t Error at ispec = " << ielement
                        << ", iz = " << 0 << ", ix = " << 0
                        << ", iprop = " << l;
                message << get_error_message(point_property_l, value_l);

                throw std::runtime_error(message.str());
              }
            }
          }
        } else if constexpr (!using_simd) {
          for (int l = 0; l < PointType::nprops; l++) {
            if (std::abs(point_property.data[l] - value_l) > 1e-6) {
              std::ostringstream message;
              message << "\n \t Error in function load_on_device";

              message << "\n \t Error at ispec = " << ielement << ", iz = " << 0
                      << ", ix = " << 0 << ", iprop = " << l;
              message << get_error_message(point_property, value_l);

              throw std::runtime_error(message.str());
            }
          }
        }
      }
    }
  }

  return;
}

void test_properties(
    specfem::compute::assembly &assembly,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const std::string &suffix) {

  auto &properties = assembly.properties;
  auto &element_types = assembly.element_types;

  //
  // ==================== HACKATHON TODO: ADD MEDIUM_TAG_ELECTROMAGNETIC_SV ===
  //

  // stage 1: check if properties are correctly constructed from the assembly
#define TEST_COMPUTE_TO_MESH(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)          \
  check_compute_to_mesh<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(assembly,  \
                                                                    mesh);

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      TEST_COMPUTE_TO_MESH,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

  // stage 2 prepare file path
  std::string output_dir = TOSTRING(TEST_OUTPUT_DIR);
  boost::filesystem::path dir_path =
      boost::filesystem::path(output_dir) / "property_io" / suffix;
  boost::filesystem::create_directories(dir_path);

  // stage 2: write properties
  specfem::IO::property_writer<specfem::IO::ASCII<specfem::IO::write> > writer(
      dir_path.string());
  writer.write(assembly);

  //
  // ==================== HACKATHON TODO: ADD MEDIUM_TAG_ELECTROMAGNETIC_SV ===
  //

  // stage 3: modify properties and check store_on_host and load_on_device
#define TEST_STORE_AND_LOAD(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)           \
  check_store_on_host<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), false>(      \
      properties, element_types);                                              \
  check_load_on_device<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), false>(     \
      properties, element_types);                                              \
  check_store_on_host<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), true>(       \
      properties, element_types);                                              \
  check_load_on_device<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG), true>(      \
      properties, element_types);

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      TEST_STORE_AND_LOAD,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef TEST_STORE_AND_LOAD

  // stage 4: restore properties to initial value from disk
  specfem::IO::property_reader<specfem::IO::ASCII<specfem::IO::read> > reader(
      dir_path.string());
  reader.read(assembly);

  //
  // ==================== HACKATHON TODO: ADD MEDIUM_TAG_ELECTROMAGNETIC_SV ===
  //

  // stage 5: check if properties are correctly written and read
  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      TEST_COMPUTE_TO_MESH,
      WHERE(DIMENSION_TAG_DIM2) WHERE(
          MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef TEST_COMPUTE_TO_MESH
  // check_compute_to_mesh<specfem::element::medium_tag::elastic_sv,
  //                       specfem::element::property_tag::isotropic>(assembly,
  //                                                                  mesh);
  // check_compute_to_mesh<specfem::element::medium_tag::elastic_sv,
  //                       specfem::element::property_tag::anisotropic>(assembly,
  //                                                                    mesh);
  // check_compute_to_mesh<specfem::element::medium_tag::acoustic,
  //                       specfem::element::property_tag::isotropic>(assembly,
  //  mesh);

  // stage 6: remove directory
  std::cout << "Removing directory: " << dir_path << std::endl;
  boost::filesystem::remove_all(dir_path);
}

TEST_F(ASSEMBLY, properties) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);
    auto suffix = std::get<4>(parameters);
    auto assembly = std::get<5>(parameters);

    try {
      test_properties(assembly, mesh, suffix);

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

  // Clear property_io directory
  std::cout << "Removing directory: " << TOSTRING(TEST_OUTPUT_DIR) << std::endl;
  std::string output_dir = TOSTRING(TEST_OUTPUT_DIR);
  boost::filesystem::path output_dir_path =
      boost::filesystem::path(output_dir) / "property_io";
  boost::filesystem::remove_all(output_dir_path);
}
