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

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd = false>
std::string get_error_message(
    const specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                     PropertyTag, false> &point_property,
    const type_real value, const int mode = 0);

template <>
std::string get_error_message(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, false> &point_property,
    const type_real value, const int mode) {
  std::ostringstream message;

  error_message_header(message, value, mode);
  message << "\t\trho = " << point_property.rho << "\n";
  message << "\t\tmu = " << point_property.mu << "\n";
  message << "\t\tkappa = " << point_property.lambdaplus2mu << "\n";

  return message.str();
}

template <>
std::string get_error_message(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, false> &point_property,
    const type_real value, const int mode) {
  std::ostringstream message;

  error_message_header(message, value, mode);
  message << "\t\trho = " << point_property.rho << "\n";
  message << "\t\tc11 = " << point_property.c11 << "\n";
  message << "\t\tc13 = " << point_property.c13 << "\n";
  message << "\t\tc15 = " << point_property.c15 << "\n";
  message << "\t\tc33 = " << point_property.c33 << "\n";
  message << "\t\tc35 = " << point_property.c35 << "\n";
  message << "\t\tc55 = " << point_property.c55 << "\n";

  return message.str();
}

template <>
std::string get_error_message(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, false> &point_property,
    const type_real value, const int mode) {
  std::ostringstream message;

  error_message_header(message, value, mode);
  message << "\t\trho_inverse = " << point_property.rho_inverse << "\n";
  message << "\t\tkappa = " << point_property.kappa << "\n";

  return message.str();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                           PropertyTag, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties);

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                           PropertyTag, false>
get_point_property(
    const int lane,
    const specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                     PropertyTag, true> &point_property);

template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic,
                           specfem::element::property_tag::isotropic, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties) {

  const auto elastic_isotropic = properties.elastic_isotropic;

  const int ispec_l = properties.h_property_index_mapping(ispec);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::isotropic, false>
      point_property;

  point_property.rho = elastic_isotropic.h_rho(ispec_l, iz, ix);
  point_property.mu = elastic_isotropic.h_mu(ispec_l, iz, ix);
  point_property.lambdaplus2mu =
      elastic_isotropic.h_lambdaplus2mu(ispec_l, iz, ix);

  return point_property;
}

template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic,
                           specfem::element::property_tag::isotropic, false>
get_point_property(
    const int lane,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, true> &point_property) {
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::isotropic, false>
      point_property_l;

  point_property_l.rho = point_property.rho[lane];
  point_property_l.mu = point_property.mu[lane];
  point_property_l.lambdaplus2mu = point_property.lambdaplus2mu[lane];

  return point_property_l;
}

template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic,
                           specfem::element::property_tag::anisotropic, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties) {

  const auto elastic_anisotropic = properties.elastic_anisotropic;

  const int ispec_l = properties.h_property_index_mapping(ispec);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::anisotropic, false>
      point_property;

  point_property.rho = elastic_anisotropic.h_rho(ispec_l, iz, ix);
  point_property.c11 = elastic_anisotropic.h_c11(ispec_l, iz, ix);
  point_property.c13 = elastic_anisotropic.h_c13(ispec_l, iz, ix);
  point_property.c15 = elastic_anisotropic.h_c15(ispec_l, iz, ix);
  point_property.c33 = elastic_anisotropic.h_c33(ispec_l, iz, ix);
  point_property.c35 = elastic_anisotropic.h_c35(ispec_l, iz, ix);
  point_property.c55 = elastic_anisotropic.h_c55(ispec_l, iz, ix);

  return point_property;
}

template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic,
                           specfem::element::property_tag::anisotropic, false>
get_point_property(
    const int lane,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, true> &point_property) {
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::anisotropic, false>
      point_property_l;

  point_property_l.rho = point_property.rho[lane];
  point_property_l.c11 = point_property.c11[lane];
  point_property_l.c13 = point_property.c13[lane];
  point_property_l.c15 = point_property.c15[lane];
  point_property_l.c33 = point_property.c33[lane];
  point_property_l.c35 = point_property.c35[lane];
  point_property_l.c55 = point_property.c55[lane];

  return point_property_l;
}

template <>
specfem::point::properties<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::acoustic,
                           specfem::element::property_tag::isotropic, false>
get_point_property(const int ispec, const int iz, const int ix,
                   const specfem::compute::properties &properties) {

  const auto acoustic_isotropic = properties.acoustic_isotropic;

  const int ispec_l = properties.h_property_index_mapping(ispec);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::acoustic,
                             specfem::element::property_tag::isotropic, false>
      point_property;

  point_property.rho_inverse =
      acoustic_isotropic.h_rho_inverse(ispec_l, iz, ix);
  point_property.kappa = acoustic_isotropic.h_kappa(ispec_l, iz, ix);

  return point_property;
}

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

  point_property_l.rho_inverse = point_property.rho_inverse[lane];
  point_property_l.kappa = point_property.kappa[lane];

  return point_property_l;
}

template <bool using_simd>
void check_eq(
    const typename specfem::datatype::simd<type_real, using_simd>::datatype &p1,
    const typename specfem::datatype::simd<type_real, using_simd>::datatype &p2,
    const int &n_simd_elements) {
  if constexpr (using_simd) {
    for (int i = 0; i < n_simd_elements; i++) {
      if (p1[i] != p2[i]) {
        std::ostringstream message;

        message << "\n \t Error in function load_on_host";
        message << "\n\t Expected: " << p1[i];
        message << "\n\t Got: " << p2[i];

        throw std::runtime_error(message.str());
      }
    }
  } else {
    if (p1 != p2) {
      std::ostringstream message;

      message << "\n \t Error in function load_on_host";
      message << "\n\t Expected: " << p1;
      message << "\n\t Got: " << p2;

      throw std::runtime_error(message.str());
    }
  }
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd>
void check_point_properties(
    const specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                     PropertyTag, using_simd> &p1,
    const specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                     PropertyTag, using_simd> &p2,
    const int &n_simd_elements);

template <bool using_simd>
void check_point_properties(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, using_simd> &p1,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, using_simd> &p2,
    const int &n_simd_elements) {
  check_eq<using_simd>(p1.rho, p2.rho, n_simd_elements);
  check_eq<using_simd>(p1.mu, p2.mu, n_simd_elements);
  check_eq<using_simd>(p1.lambdaplus2mu, p2.lambdaplus2mu, n_simd_elements);
  check_eq<using_simd>(p1.lambda,
                       p2.lambdaplus2mu -
                           (static_cast<typename specfem::datatype::simd<
                                type_real, using_simd>::datatype>(2.0)) *
                               p2.mu,
                       n_simd_elements);
  check_eq<using_simd>(p1.rho_vp, Kokkos::sqrt(p2.rho * p2.lambdaplus2mu),
                       n_simd_elements);
  check_eq<using_simd>(p1.rho_vs, Kokkos::sqrt(p2.rho * p2.mu),
                       n_simd_elements);
}

template <bool using_simd>
void check_point_properties(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, using_simd> &p1,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, using_simd> &p2,
    const int &n_simd_elements) {
  check_eq<using_simd>(p1.rho, p2.rho, n_simd_elements);
  check_eq<using_simd>(p1.c11, p2.c11, n_simd_elements);
  check_eq<using_simd>(p1.c13, p2.c13, n_simd_elements);
  check_eq<using_simd>(p1.c15, p2.c15, n_simd_elements);
  check_eq<using_simd>(p1.c33, p2.c33, n_simd_elements);
  check_eq<using_simd>(p1.c35, p2.c35, n_simd_elements);
  check_eq<using_simd>(p1.c55, p2.c55, n_simd_elements);
  check_eq<using_simd>(p1.rho_vp, Kokkos::sqrt(p2.rho * p2.c33),
                       n_simd_elements);
  check_eq<using_simd>(p1.rho_vs, Kokkos::sqrt(p2.rho * p2.c55),
                       n_simd_elements);
}

template <bool using_simd>
void check_point_properties(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, using_simd> &p1,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, using_simd> &p2,
    const int &n_simd_elements) {
  check_eq<using_simd>(p1.rho_inverse, p2.rho_inverse, n_simd_elements);
  check_eq<using_simd>(p1.kappa, p2.kappa, n_simd_elements);
  check_eq<using_simd>(
      p1.kappa_inverse,
      (static_cast<
          typename specfem::datatype::simd<type_real, using_simd>::datatype>(
          1.0)) /
          p2.kappa,
      n_simd_elements);
  check_eq<using_simd>(p1.rho_vpinverse,
                       Kokkos::sqrt(p2.rho_inverse * p2.kappa_inverse),
                       n_simd_elements);
}

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

  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();
  using PointType =
      specfem::point::properties<specfem::dimension::type::dim2, MediumTag,
                                 PropertyTag, using_simd>;
  const int nprops = PointType::nprops;

  for (int i = 0; i < ispecs.extent(0); ++i) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        const int ielement = ispecs(i);
        const int n_simd_elements = (simd_size + ielement > elements.size())
                                        ? elements.size() - ielement
                                        : simd_size;
        for (int j = 0; j < n_simd_elements; j++) {
          auto point_property = get_point_property<MediumTag, PropertyTag>(
              ielement + j, iz, ix, properties);
          point_property.compute();
          const type_real value = values_to_store(i);
          for (int iprop = 0; iprop < nprops; iprop++) {
            if (point_property.data[iprop] != value) {
              std::ostringstream message;

              message << "\n \t Error at ispec = " << ielement + j
                      << ", iz = " << iz << ", ix = " << ix;
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
        auto point_property = get_point_property<MediumTag, PropertyTag>(
            ielement, iz, ix, properties);
        point_property.compute();
        const int ispec_mesh = mapping.compute_to_mesh(ielement);
        auto material =
            std::get<specfem::medium::material<MediumTag, PropertyTag> >(
                materials[ispec_mesh]);
        auto value = material.get_properties();
        if (point_property != value) {
          std::ostringstream message;

          message << "\n \t Error at ispec = " << ielement << ", iz = " << iz
                  << ", ix = " << ix;

          message << get_error_message(value, 0.0, 1);
          message << get_error_message(point_property, 0.0, 2);

          throw std::runtime_error(message.str());
        }
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

  constexpr int nprops = PointType::nprops;
  using value_type = typename PointType::value_type;
  value_type data[nprops];

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
        for (int iprop = 0; iprop < nprops; iprop++) {
          data[iprop] = value;
        }
        PointType point(data);
        PointType point_loaded;
        specfem::compute::store_on_host(index, properties, point);
        specfem::compute::load_on_host(index, properties, point_loaded);
        check_point_properties<using_simd>(point, point_loaded,
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
  constexpr int nprops = PointType::nprops;
  using value_type = typename PointType::value_type;

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

  Kokkos::View<value_type ***[N], Kokkos::DefaultExecutionSpace>
      point_properties("point_properties", ngllz, ngllx, nprops);
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
        for (int iprop = 0; iprop < nprops; iprop++) {
          point_properties(iz, ix, iprop, i) = point.data[iprop];
        }
      });

  Kokkos::fence();
  Kokkos::deep_copy(h_point_properties, point_properties);

  for (int i = 0; i < N; i++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        using simd = specfem::datatype::simd<type_real, using_simd>;
        value_type data[nprops];
        for (int iprop = 0; iprop < nprops; iprop++) {
          data[iprop] = h_point_properties(iz, ix, iprop, i);
        }
        const PointType point_property = { data };
        const int ielement = ispecs_h(i);
        constexpr int simd_size = PointType::simd::size();
        const int n_simd_elements = (simd_size + ielement > element_size)
                                        ? element_size - ielement
                                        : simd_size;
        const type_real value_l = values_to_store_h(i);
        for (int lane = 0; lane < n_simd_elements; lane++) {
          for (int iprop = 0; iprop < nprops; iprop++) {
            if constexpr (using_simd) {
              const auto point_property_l =
                  get_point_property(lane, point_property);
              if (point_property_l.data[iprop] != value_l) {
                std::ostringstream message;

                message << "\n \t Error in function load_on_device";

                message << "\n \t Error at ispec = " << ielement
                        << ", iz = " << 0 << ", ix = " << 0;
                message << get_error_message(point_property_l, value_l);

                throw std::runtime_error(message.str());
              }
            } else if constexpr (!using_simd) {
              if (point_property.data[iprop] != value_l) {
                std::ostringstream message;
                message << "\n \t Error in function load_on_device";

                message << "\n \t Error at ispec = " << ielement
                        << ", iz = " << 0 << ", ix = " << 0;
                message << get_error_message(point_property, value_l);

                throw std::runtime_error(message.str());
              }
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
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh) {

  auto &properties = assembly.properties;
  auto &element_types = assembly.element_types;

  // stage 1: check if properties are correctly constructed from the assembly
#define TEST_COMPUTE_TO_MESH(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)          \
  check_compute_to_mesh<GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(assembly,  \
                                                                    mesh);

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      TEST_COMPUTE_TO_MESH,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

  // stage 2: write properties
  specfem::IO::property_writer<specfem::IO::ASCII<specfem::IO::write> > writer(
      ".");
  writer.write(assembly);

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
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef TEST_STORE_AND_LOAD

  // stage 4: restore properties to initial value from disk
  specfem::IO::property_reader<specfem::IO::ASCII<specfem::IO::read> > reader(
      ".");
  reader.read(assembly);

  // stage 5: check if properties are correctly written and read
  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      TEST_COMPUTE_TO_MESH,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC)
          WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef TEST_COMPUTE_TO_MESH
  // check_compute_to_mesh<specfem::element::medium_tag::elastic,
  //                       specfem::element::property_tag::isotropic>(assembly,
  //                                                                  mesh);
  // check_compute_to_mesh<specfem::element::medium_tag::elastic,
  //                       specfem::element::property_tag::anisotropic>(assembly,
  //                                                                    mesh);
  // check_compute_to_mesh<specfem::element::medium_tag::acoustic,
  //                       specfem::element::property_tag::isotropic>(assembly,
  //  mesh);
}

TEST_F(ASSEMBLY, properties) {
  for (auto parameters : *this) {
    auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);
    auto assembly = std::get<4>(parameters);

    try {
      test_properties(assembly, mesh);

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
