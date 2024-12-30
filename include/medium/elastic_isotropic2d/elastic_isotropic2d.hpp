#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "globals.h"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <>
class properties<specfem::element::medium_tag::elastic,
                 specfem::element::property_tag::isotropic> {
public:
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension of the material
  constexpr static auto medium_tag =
      specfem::element::medium_tag::elastic; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic; ///< Property tag

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Construct a new elastic isotropic material
   *
   * @param density Density of the material
   * @param cs Shear wave speed
   * @param cp Compressional wave speed
   * @param Qkappa Attenuation factor for bulk modulus
   * @param Qmu Attenuation factor for shear modulus
   * @param compaction_grad Compaction gradient
   */
  properties(const type_real &density, const type_real &cs, const type_real &cp,
             const type_real &Qkappa, const type_real &Qmu,
             const type_real &compaction_grad)
      : density(density), cs(cs), cp(cp), Qkappa(Qkappa), Qmu(Qmu),
        compaction_grad(compaction_grad), lambdaplus2mu(density * cp * cp),
        mu(density * cs * cs), lambda(lambdaplus2mu - 2.0 * mu),
        kappa(lambda + mu), young(9.0 * kappa * mu / (3.0 * kappa + mu)),
        poisson(0.5 * (cp * cp - 2.0 * cs * cs) / (cp * cp - cs * cs)) {
    if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
      std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }

    if (this->poisson < -1.0 || this->poisson > 0.5)
      std::runtime_error("Poisson's ratio out of range");
  };

  /**
   * @brief Default constructor
   *
   */
  properties() = default;

  ///@}

  /**
   * @brief Check if 2 materials have the same properties
   *
   * @param other Material to compare with
   * @return true If the materials have the same properties
   */
  bool operator==(const properties<specfem::element::medium_tag::elastic,
                                   specfem::element::property_tag::isotropic>
                      &other) const {

    return (std::abs(this->density - other.density) < 1e-6 &&
            std::abs(this->cp - other.cp) < 1e-6 &&
            std::abs(this->cs - other.cs) < 1e-6 &&
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
  bool operator!=(const properties<specfem::element::medium_tag::elastic,
                                   specfem::element::property_tag::isotropic>
                      &other) const {
    return !(*this == other);
  }

  /**
   * @brief Get the material properties
   *
   * @return specfem::point::properties Material properties
   */
  inline specfem::point::properties<dimension, medium_tag, property_tag, false>
  get_properties() const {
    return { this->lambdaplus2mu, this->mu, this->density };
  }

  inline std::string print() const {
    std::ostringstream message;

    message << "- Elastic Material : \n"
            << "    Properties:\n"
            << "      density : " << this->density << "\n"
            << "      cs : " << this->cs << "\n"
            << "      cp : " << this->cp << "\n"
            << "      kappa : " << this->kappa << "\n"
            << "      mu : " << this->mu << "\n"
            << "      Qkappa : " << this->Qkappa << "\n"
            << "      Qmu : " << this->Qmu << "\n"
            << "      lambda : " << this->lambda << "\n"
            << "      mu : " << this->mu << "\n"
            << "      youngs modulus : " << this->young << "\n"
            << "      poisson ratio : " << this->poisson << "\n";

    return message.str();
  }

private:
  type_real density;         ///< Density of the material
  type_real cs;              ///< Shear wave speed
  type_real cp;              ///< Compressional wave speed
  type_real Qkappa;          ///< Attenuation factor for bulk modulus
  type_real Qmu;             ///< Attenuation factor for shear modulus
  type_real compaction_grad; ///< Compaction gradient
  type_real lambdaplus2mu;   ///< Lame parameter
  type_real mu;              ///< Lame parameter
  type_real lambda;          ///< Lame parameter
  type_real kappa;           ///< Bulk modulus
  type_real young;           ///< Young's modulus
  type_real poisson;         ///< Poisson's ratio
};

template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION specfem::point::stress<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    UseSIMD>
impl_compute_stress(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        UseSIMD> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_zz, sigma_xz;

  // P_SV case
  // sigma_xx
  sigma_xx = properties.lambdaplus2mu * du(0, 0) + properties.lambda * du(1, 1);

  // sigma_zz
  sigma_zz = properties.lambdaplus2mu * du(1, 1) + properties.lambda * du(0, 0);

  // sigma_xz
  sigma_xz = properties.mu * (du(0, 1) + du(1, 0));

  specfem::datatype::VectorPointViewType<type_real, 2, 2, UseSIMD> T;

  T(0, 0) = sigma_xx;
  T(0, 1) = sigma_xz;
  T(1, 0) = sigma_xz;
  T(1, 1) = sigma_zz;

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
                                 specfem::element::medium_tag::elastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointPropertiesType &properties,
    const AdjointPointFieldType &adjoint_field,
    const BackwardPointFieldType &backward_field,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  static_assert(specfem::globals::simulation_wave == specfem::wave::p_sv ||
                    specfem::globals::simulation_wave == specfem::wave::sh,
                "Only P-SV and SH waves are supported.");

  const auto kappa = properties.lambdaplus2mu - properties.mu;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {

    /*
    In the following the explanation for the SH wave kernels is given.

    Computing the deviatoric strain tensor for SH waves:

                           strain epsilon
      D = [    dux_dx         1/2(dux_dy+duy_dx) 1/2(dux_dz+duz_dx),
          1/2(duy_dx+dux_dy)     duy_dy         1/2(duy_dz+duz_dy),
          1/2(duz_dx+dux_dz) 1/2(duz_dy+duy_dz)     duz_dz   ]

                                trace of strain diagonalized
           [ 1/3 (dux_dx + duy_dy + duz_dz)       0                  0,
        -    0             1/3 (dux_dx + duy_dy + duz_dz)            0,
             0                    0     1/3 (dux_dx + duy_dy + duz_dx)  ]

    We use `s` as the "standard" wavefield and `s#` as the adjoint wavefield.
    We use `ad_` as prefix for the adjoint strainfield, and `b_` as prefix for
    the "standard" strainfield.
    */

    // Compute the gradient of the adjoint field
    // ad_dsxx = 0.5 * (ds#x/dx + ds#x/dx) = ds#x/dx
    const auto ad_dsxx = adjoint_derivatives.du(0, 0);

    // ad_dsxz = 0.5 * (ds#x/dz + ds#z/dx)
    const auto ad_dsxz =
        static_cast<type_real>(0.5) *
        (adjoint_derivatives.du(0, 1) + adjoint_derivatives.du(1, 0));

    // ad_dszz = 0.5 * (ds#z/dz + ds#z/dz) = ds#z/dz
    const auto ad_dszz = adjoint_derivatives.du(1, 1);

    // Compute the gradient of the backward field
    // b_dsxx = 0.5 * (dsx/dx + dsx/dx) = dsx/dx
    const auto b_dsxx = backward_derivatives.du(0, 0);

    // b_dsxz = 0.5 * (dsx/dz + dsz/dx)
    const auto b_dsxz =
        static_cast<type_real>(0.5) *
        (backward_derivatives.du(0, 1) + backward_derivatives.du(1, 0));

    // b_dszz = 0.5 * (dsz/dz + dsz/dz) = dsz/dz
    const auto b_dszz = backward_derivatives.du(1, 1);

    // what's this?
    // --------------------------------------
    // const type_real kappa_kl =
    //     -1.0 * kappa * dt * ((ad_dsxx + ad_dszz) * (b_dsxx + b_dszz));
    // const type_real mu_kl = -2.0 * properties.mu * dt *
    //                         (ad_dsxx * b_dsxx + ad_dszz * b_dszz +
    //                          2.0 * ad_dsxz * b_dsxz - 1.0 / 3.0 * kappa_kl);
    // const type_real rho_kl =
    //     -1.0 * properties.rho * dt *
    //     (specfem::algorithms::dot(adjoint_field.acceleration,
    //                               backward_field.displacement));
    // --------------------------------------

    // In the papers we use dagger for the notation of the adjoint wavefield
    // here I'm using #

    // Part of Tromp et al. 2005, Eq 18
    // div(s#) * div(s)
    auto kappa_kl = (ad_dsxx + ad_dszz) * (b_dsxx + b_dszz);

    // Part of Tromp et al. 2005, Eq 17
    // [eps+ : eps] - 1/3 [div (s#) * div(s)]
    // I am not clear on how we get to the following form but from the
    // GPU cuda code from the fortran code I assume that there is an
    // assumption being made that eps#_i * eps_j = eps#_j * eps_i in the
    // isotropic case due to the symmetry of the voigt notation stiffness
    // matrix. Since x
    auto mu_kl = (ad_dsxx * b_dsxx + ad_dszz * b_dszz +
                  static_cast<type_real>(2.0) * ad_dsxz * b_dsxz -
                  static_cast<type_real>(1.0 / 3.0) * kappa_kl);

    // This notation/naming is confusing with respect to the physics.
    // Should be forward.acceleration dotted with adjoint displacement
    // See Tromp et al. 2005, Equation 14.
    auto rho_kl = specfem::algorithms::dot(adjoint_field.acceleration,
                                           backward_field.displacement);

    // Finishing the kernels
    kappa_kl = static_cast<type_real>(-1.0) * kappa * dt * kappa_kl;
    mu_kl = static_cast<type_real>(-2.0) * properties.mu * dt * mu_kl;
    rho_kl = static_cast<type_real>(-1.0) * properties.rho * dt * rho_kl;

    // rho' kernel, first term in Equation 20
    const auto rhop_kl = rho_kl + kappa_kl + mu_kl;

    // beta (shear wave) kernel, second term in Equation 20
    const auto beta_kl = static_cast<type_real>(2.0) *
                         (mu_kl - static_cast<type_real>(4.0 / 3.0) *
                                      properties.mu / kappa * kappa_kl);

    // alpha (compressional wave) kernel, third and last term in Eq. 20
    // of Tromp et al 2005.
    const auto alpha_kl =
        static_cast<type_real>(2.0) *
        (static_cast<type_real>(1.0) +
         static_cast<type_real>(4.0 / 3.0) * properties.mu / kappa) *
        kappa_kl;

    return { rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl, beta_kl };
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {

    /*
    SH (membrane) waves
    -------------------

    The deviatroic strain tensor for SH waves in and isotropic elastic medium
    is given by:

    SH-waves: plane strain assumption ux==uz==0 and d/dy==0
      D = [   0             1/2 duy_dx       0,                   [0 0 0,
            1/2 duy_dx       0             1/2 duy_dz,       -     0 0 0,
              0             1/2 duy_dz       0          ]          0 0 0]

    Resulting in the following kernels D# : D
    D# : D = sum_i sum_j D#_ij * D_ij
              = 1/2du#y_dx * 1/2duy_dx + 1/2du#y_dx * 1/2duy_dx
                  + 1/2du#y_dz * 1/2duy_dz + 1/2du#y_dz * 1/2duy_dz
              = 1/2 ( du#y_dx * duy_dx) + 1/2 (du#y_dz * duy_dz)
              = 1/2 ( du#y_dx * duy_dx + du#y_dz * duy_dz )

    */
    const auto mu_kl =
        static_cast<type_real>(-2.0) * properties.mu * dt *
        static_cast<type_real>(0.5) *
        // du#y_dx * duy_dx +
        (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
         // du#y_dz * duy_dz
         adjoint_derivatives.du(1, 0) * backward_derivatives.du(1, 0));
    const auto rho_kl = static_cast<type_real>(-1.0) * properties.rho * dt *
                        specfem::algorithms::dot(adjoint_field.acceleration,
                                                 backward_field.displacement);
    const auto kappa_kl = decltype(mu_kl)(0.0);

    const auto rhop_kl = rho_kl + kappa_kl + mu_kl;
    const auto alpha_kl = 0.0;
    const auto beta_kl = static_cast<type_real>(2.0) * mu_kl;

    return { rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl, beta_kl };
  }
}

template <typename MemberType, typename IteratorType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const MemberType &team, const IteratorType &iterator,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &field,
    const specfem::wavefield::type wavefield_component,
    WavefieldViewType wavefield) {

  using FieldDerivativesType =
      specfem::point::field_derivatives<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::elastic,
                                        false>;

  using PointPropertyType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
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
      return field.displacement;
    } else {
      Kokkos::abort("component not supported");
    }
  }();

  if (wavefield_component == specfem::wavefield::type::pressure) {

    specfem::algorithms::gradient(
        team, iterator, assembly.partial_derivatives, quadrature.hprime_gll,
        active_field,
        [&](const typename IteratorType::index_type &iterator_index,
            const FieldDerivativesType::ViewType &du) {
          const auto &index = iterator_index.index;
          PointPropertyType point_property;

          specfem::compute::load_on_device(index, properties, point_property);

          // P_SV case
          // sigma_xx
          const auto sigma_xx = point_property.lambdaplus2mu * du(0, 0) +
                                point_property.lambda * du(1, 1);

          // sigma_zz
          const auto sigma_zz = point_property.lambdaplus2mu * du(1, 1) +
                                point_property.lambda * du(0, 0);

          // sigma_yy
          const auto sigma_yy =
              point_property.lambdaplus2mu * (du(0, 0) + du(1, 1));

          wavefield(index.ispec, index.iz, index.ix, 0) =
              -1.0 * (sigma_xx + sigma_zz + sigma_yy) / 3.0;
        });

    return;
  }

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, iterator.chunk_size()), [&](const int &i) {
        const auto iterator_index = iterator(i);
        const auto &index = iterator_index.index;
        wavefield(index.ispec, index.iz, index.ix, 0) =
            active_field(iterator_index.ielement, index.iz, index.ix, 0);
        wavefield(index.ispec, index.iz, index.ix, 1) =
            active_field(iterator_index.ielement, index.iz, index.ix, 1);
      });

  return;
}

} // namespace medium
} // namespace specfem
