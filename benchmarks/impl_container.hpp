#pragma once

#include "compute/element_types/element_types.hpp"
#include "compute/impl/value_containers.hpp"
#include "enumerations/medium.hpp"
#include "impl_point.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace benchmarks {

#define DEFINE_CONTAINER(prop, i)                                              \
  KOKKOS_INLINE_FUNCTION type_real &prop(const int &ispec, const int &iz,      \
                                         const int &ix) const {                \
    return Base::data(ispec, iz, ix, i);                                       \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION type_real &h_##prop(const int &ispec, const int &iz,  \
                                             const int &ix) const {            \
    return Base::h_data(ispec, iz, ix, i);                                     \
  }

namespace impl {
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int N>
struct impl_properties_container {
  using view_type = typename Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                          Kokkos::DefaultExecutionSpace>;
  constexpr static auto nprops = N;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;

  int nspec; ///< total number of acoustic spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension

  view_type data;
  view_type::HostMirror h_data;

  impl_properties_container() = default;

  impl_properties_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        data("specfem::benchmarks::properties::data", nspec, ngllz, ngllx, N),
        h_data(Kokkos::create_mirror_view(data)) {}

private:
  template <typename PointProperties>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_properties(const specfem::point::index<dimension> &index,
                  PointProperties &property, const view_type &target) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == medium_tag,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_tag,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    for (int i = 0; i < nprops; i++) {
      property.data[i] = target(ispec, iz, ix, i);
    }

    property.compute();
  }

public:
  template <typename PointProperties>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_device_properties(const specfem::point::index<dimension> &index,
                         PointProperties &property) const {
    load_properties(index, property, data);
  }

  template <typename PointProperties>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_host_properties(const specfem::point::index<dimension> &index,
                       PointProperties &property) const {
    load_properties(index, property, h_data);
  }

  template <typename PointProperties>
  inline void assign(const specfem::point::index<dimension> &index,
                     const PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == medium_tag,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_tag,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    for (int i = 0; i < nprops; i++) {
      h_data(ispec, iz, ix, i) = property.data[i];
    }
  }

  void copy_to_device() { Kokkos::deep_copy(data, h_data); }

  void copy_to_host() { Kokkos::deep_copy(h_data, data); }
};
} // namespace impl

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
struct properties_container {
  static_assert("Material type not implemented");
};

template <>
struct properties_container<specfem::element::medium_tag::acoustic,
                            specfem::element::property_tag::isotropic>
    : public impl::impl_properties_container<
          specfem::element::medium_tag::acoustic,
          specfem::element::property_tag::isotropic, 2> {
  using Base =
      impl::impl_properties_container<specfem::element::medium_tag::acoustic,
                                      specfem::element::property_tag::isotropic,
                                      2>;
  using Base::Base;

  DEFINE_CONTAINER(rho_inverse, 0)
  DEFINE_CONTAINER(kappa, 1)
};

template <>
struct properties_container<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::isotropic>
    : public impl::impl_properties_container<
          specfem::element::medium_tag::elastic,
          specfem::element::property_tag::isotropic, 3> {
  using Base =
      impl::impl_properties_container<specfem::element::medium_tag::elastic,
                                      specfem::element::property_tag::isotropic,
                                      3>;
  using Base::Base;

  DEFINE_CONTAINER(lambdaplus2mu, 0)
  DEFINE_CONTAINER(mu, 1)
  DEFINE_CONTAINER(rho, 2)
};

template <>
struct properties_container<specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::anisotropic>
    : public impl::impl_properties_container<
          specfem::element::medium_tag::elastic,
          specfem::element::property_tag::anisotropic, 10> {
  using Base = impl::impl_properties_container<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic, 10>;
  using Base::Base;

  DEFINE_CONTAINER(c11, 0)
  DEFINE_CONTAINER(c13, 1)
  DEFINE_CONTAINER(c15, 2)
  DEFINE_CONTAINER(c33, 3)
  DEFINE_CONTAINER(c35, 4)
  DEFINE_CONTAINER(c55, 5)
  DEFINE_CONTAINER(c12, 6)
  DEFINE_CONTAINER(c23, 7)
  DEFINE_CONTAINER(c25, 8)
  DEFINE_CONTAINER(rho, 9)
};

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
struct material_properties
    : public specfem::benchmarks::properties_container<type, property> {
  constexpr static auto value_type = type;
  constexpr static auto property_type = property;
  constexpr static auto dimension = specfem::dimension::type::dim2;

  material_properties() = default;

  material_properties(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int ngllx,
      const specfem::mesh::materials &materials, const bool has_gll_model,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : specfem::benchmarks::properties_container<type, property>(
            elements.extent(0), ngllz, ngllx) {

    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      property_index_mapping(ispec) = count;
      if (!has_gll_model) {
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int ix = 0; ix < ngllx; ++ix) {
            // Get the material at index from mesh::materials
            auto material =
                std::get<specfem::medium::material<type, property> >(
                    materials[ispec]);

            // Assign the material property to the property container
            specfem::benchmarks::properties<dimension, value_type,
                                            property_type, false>
                point_property = material.get_benchmark_properties().data();
            this->assign(specfem::point::index<dimension>(count, iz, ix),
                         point_property);
          }
        }
      }
      count++;
    }

    if (!has_gll_model) {
      this->copy_to_device();
    }

    return;
  }
};

struct assembly_properties : public specfem::compute::impl::value_containers<
                                 specfem::benchmarks::material_properties> {
  /**
   * @name Constructors
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  assembly_properties() = default;

  /**
   * @brief Construct a new properties object from mesh information
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of quadrature points in z direction
   * @param ngllx Number of quadrature points in x direction
   * @param mapping Mapping of spectral element index from mesh to assembly
   * @param tags Element Tags for every spectral element
   * @param materials Material properties for every spectral element
   * @param has_gll_model Whether a GLL model is present (skip material property
   * assignment if true)
   */
  assembly_properties(const int nspec, const int ngllz, const int ngllx,
                      const specfem::compute::element_types &element_types,
                      const specfem::mesh::materials &materials,
                      bool has_gll_model) {

    this->nspec = nspec;
    this->ngllz = ngllz;
    this->ngllx = ngllx;

    this->property_index_mapping =
        Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(
            "specfem::compute::properties::property_index_mapping", nspec);
    this->h_property_index_mapping =
        Kokkos::create_mirror_view(property_index_mapping);

    const auto elastic_isotropic_elements = element_types.get_elements_on_host(
        specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic);

    const auto elastic_anisotropic_elements =
        element_types.get_elements_on_host(
            specfem::element::medium_tag::elastic,
            specfem::element::property_tag::anisotropic);

    const auto acoustic_elements = element_types.get_elements_on_host(
        specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic);

    for (int ispec = 0; ispec < nspec; ++ispec) {
      h_property_index_mapping(ispec) = -1;
    }

    acoustic_isotropic = specfem::benchmarks::material_properties<
        specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic>(
        acoustic_elements, ngllz, ngllx, materials, has_gll_model,
        h_property_index_mapping);

    elastic_isotropic = specfem::benchmarks::material_properties<
        specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic>(
        elastic_isotropic_elements, ngllz, ngllx, materials, has_gll_model,
        h_property_index_mapping);

    elastic_anisotropic = specfem::benchmarks::material_properties<
        specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic>(
        elastic_anisotropic_elements, ngllz, ngllx, materials, has_gll_model,
        h_property_index_mapping);

    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
  }

  ///@}

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    specfem::compute::impl::value_containers<
        specfem::benchmarks::material_properties>::copy_to_host();
  }

  void copy_to_device() {
    specfem::compute::impl::value_containers<
        specfem::benchmarks::material_properties>::copy_to_device();
  }
};

} // namespace benchmarks
} // namespace specfem
