#pragma once

#include "algorithms/dot.hpp"
#include "algorithms/gradient.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/specfem_enums.hpp"
#include "medium/properties.hpp"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <exception>
#include <iostream>
#include <ostream>
#include <sstream>

namespace specfem {
namespace medium {

/**
 * @brief Template specialization for acoustic isotropic material properties
 *
 */
template <>
class properties<specfem::element::medium_tag::acoustic,
                 specfem::element::property_tag::isotropic> {

public:
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension of the material
  constexpr static auto medium_tag =
      specfem::element::medium_tag::acoustic; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic; ///< Property tag

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Construct a new acoustic isotropic material
   *
   * @param density Density of the material
   * @param cp Compressional wave speed
   * @param Qkappa Attenuation factor for bulk modulus
   * @param Qmu Attenuation factor for shear modulus
   * @param compaction_grad Compaction gradient
   */
  properties(const type_real &density, const type_real &cp,
             const type_real &Qkappa, const type_real &Qmu,
             const type_real &compaction_grad)
      : density(density), cp(cp), Qkappa(Qkappa), Qmu(Qmu),
        compaction_grad(compaction_grad), lambdaplus2mu(density * cp * cp),
        lambda(lambdaplus2mu), kappa(lambda) {
    if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
      std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }
  }

  /**
   * @brief Check if 2 materials have the same properties
   *
   * @param other Material to compare with
   * @return true If the materials have the same properties
   */
  bool operator==(const properties<specfem::element::medium_tag::acoustic,
                                   specfem::element::property_tag::isotropic>
                      &other) const {

    return (std::abs(this->density - other.density) < 1e-6 &&
            std::abs(this->cp - other.cp) < 1e-6 &&
            std::abs(this->Qkappa - other.Qkappa) < 1e-6 &&
            std::abs(this->Qmu - other.Qmu) < 1e-6 &&
            std::abs(this->compaction_grad - other.compaction_grad) < 1e-6);
  }

  /**
   * @brief Check if 2 materials have different properties
   *
   * @param other Material to compare with
   * @return true If the materials have different properties
   */
  bool operator!=(const properties<specfem::element::medium_tag::acoustic,
                                   specfem::element::property_tag::isotropic>
                      &other) const {
    return !(*this == other);
  }

  /**
   * @brief Default constructor
   *
   */
  properties() = default;
  ///@}

  ~properties() = default;

  /**
   * @brief Get the properties of the material
   *
   * @return specfem::point::properties Material properties
   */
  inline specfem::point::properties<dimension, medium_tag, property_tag, false>
  get_properties() const {
    return { static_cast<type_real>(1.0) /
                 static_cast<type_real>(lambdaplus2mu),
             static_cast<type_real>(1.0) / static_cast<type_real>(density),
             this->kappa };
  }

  inline std::string print() const {
    std::ostringstream message;

    message << "- Acoustic Material : \n"
            << "    Properties:\n"
            << "      density : " << this->density << "\n"
            << "      cp : " << this->cp << "\n"
            << "      kappa : " << this->kappa << "\n"
            << "      Qkappa : " << this->Qkappa << "\n"
            << "      lambda : " << this->lambda << "\n"
            << "      youngs modulus : 0.0 \n"
            << "      poisson ratio :  0.5 \n";

    return message.str();
  }

private:
  type_real density;         ///< Density of the material
  type_real cp;              ///< Compressional wave speed
  type_real Qkappa;          ///< Attenuation factor for bulk modulus
  type_real Qmu;             ///< Attenuation factor for shear modulus
  type_real compaction_grad; ///< Compaction gradient
  type_real lambdaplus2mu;   ///< Lame parameter
  type_real lambda;          ///< Lame parameter
  type_real kappa;           ///< Bulk modulus
};

template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION specfem::point::stress<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    UseSIMD>
impl_compute_stress(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        UseSIMD> &field_derivatives) {

  const auto &du = field_derivatives.du;

  specfem::datatype::VectorPointViewType<type_real, 2, 1, UseSIMD> T;

  T(0, 0) = properties.rho_inverse * du(0, 0);
  T(1, 0) = properties.rho_inverse * du(1, 0);

  return { T };
}

template <typename PointPropertiesType, typename AdjointPointFieldType,
          typename BackwardPointFieldType, typename PointFieldDerivativesType>
KOKKOS_FUNCTION specfem::point::kernels<
    PointPropertiesType::dimension, PointPropertiesType::medium_tag,
    PointPropertiesType::property_tag, PointPropertiesType::simd::using_simd>
impl_compute_frechet_derivatives(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointPropertiesType &properties,
    const AdjointPointFieldType &adjoint_field,
    const BackwardPointFieldType &backward_field,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  const auto rho_kl =
      (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
       adjoint_derivatives.du(1, 0) * backward_derivatives.du(1, 0)) *
      properties.rho_inverse * dt;

  const auto kappa_kl = specfem::algorithms::dot(adjoint_field.acceleration,
                                                 backward_field.displacement) *
                        static_cast<type_real>(1.0) / properties.kappa * dt;

  return { rho_kl, kappa_kl };
}

template <typename MemberType, typename IteratorType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &field,
    const specfem::wavefield::type wavefield_component,
    WavefieldViewType wavefield) {

  using FieldDerivativesType =
      specfem::point::field_derivatives<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::acoustic,
                                        false>;

  using PointPropertyType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, false>;

  const auto &properties = assembly.properties;

  const auto &active_field = [&]() {
    if (wavefield_component == specfem::wavefield::type::displacement) {
      return field.displacement;
    } else if (wavefield_component == specfem::wavefield::type::velocity) {
      return field.velocity;
    } else if (wavefield_component == specfem::wavefield::type::acceleration) {
      return field.acceleration;
    } else if (wavefield_component == specfem::wavefield::type::pressure) {
      return field.acceleration;
    } else {
      Kokkos::abort("component not supported");
    }
  }();

  if (wavefield_component == specfem::wavefield::type::pressure) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, iterator.chunk_size()),
                         [&](const int &i) {
                           const auto iterator_index = iterator(i);
                           const auto index = iterator_index.index;
                           wavefield(index.ispec, index.iz, index.ix, 0) =
                               -1.0 * active_field(iterator_index.ielement,
                                                   index.iz, index.ix, 0);
                         });

    return;
  }

  specfem::algorithms::gradient(
      team, iterator, assembly.partial_derivatives, quadrature.hprime_gll,
      active_field,
      [&](const typename IteratorType::index_type &iterator_index,
          const FieldDerivativesType::ViewType &du) {
        const auto &index = iterator_index.index;
        PointPropertyType point_property;

        specfem::compute::load_on_device(index, properties, point_property);

        FieldDerivativesType point_field_derivatives(du);

        const auto point_stress =
            impl_compute_stress(point_property, point_field_derivatives);

        wavefield(index.ispec, index.iz, index.ix, 0) = point_stress.T(0, 0);
        wavefield(index.ispec, index.iz, index.ix, 1) = point_stress.T(1, 0);
      });

  return;
}

} // namespace medium
} // namespace specfem
