#pragma once

#include "enumerations/interface.hpp"
#include "impl/compute_coupling.hpp"
#include "impl/compute_mass_matrix.hpp"
#include "impl/compute_seismogram.hpp"
#include "impl/compute_source_interaction.hpp"
#include "impl/compute_stiffness_interaction.hpp"
#include "impl/divide_mass_matrix.hpp"
#include "impl/invert_mass_matrix.hpp"

namespace specfem {
namespace kokkos_kernels {

/**
 * @brief Class to compute the domain kernels for the simulation
 *
 * This class computes the domain kernels for the simulation. It is a template
 * class that takes the wavefield type, dimension tag, and number of GLL points
 * as template parameters.
 *
 * @tparam WavefieldType Type of the wavefield (e.g., elastic, acoustic)
 * @tparam DimensionTag Dimension tag (e.g., 2D, 3D)
 * @tparam NGLL Number of GLL points
 *
 */
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionTag, int NGLL>
class domain_kernels {
public:
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto wavefield = WavefieldType;
  constexpr static auto ngll = NGLL;

  /**
   * @brief Constructor for the domain_kernels class
   *
   * This constructor initializes the domain_kernels class with the given
   * assembly object.
   *
   * @param assembly The assembly object containing the mesh and other
   * information
   *
   * @note The constructor initializes the coupling interfaces for 2D elastic
   * and acoustic media.
   *
   */
  domain_kernels(const specfem::assembly::assembly<dimension_tag> &assembly)
      : assembly(assembly) {}

  /**
   * @brief Updates the wavefield for a given MediumTag
   *
   * This function updates the wavefield for a given MediumTag type. It computes
   * the coupling, source interaction, stiffness interaction, and divides the
   * mass matrix. The function is specialized for different MediumTag types and
   *
   * @tparam MediumTag Medium for which the wacefield is updated
   * @param istep Time step for which the wavefield is updated
   * @return int Number of elements updated
   */
  template <specfem::element::medium_tag MediumTag>
  inline int update_wavefields(const int istep) {

    int elements_updated = 0;

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
         INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
         BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                      COMPOSITE_STACEY_DIRICHLET)),
        {
          constexpr auto self_medium =
              specfem::interface::attributes<_dimension_tag_,
                                             _interface_tag_>::self_medium();
          if constexpr (dimension_tag == _dimension_tag_ &&
                        self_medium == MediumTag) {
            impl::compute_coupling<_dimension_tag_, _connection_tag_, wavefield,
                                   _interface_tag_, _boundary_tag_>(assembly);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
         BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                      COMPOSITE_STACEY_DIRICHLET)),
        {
          if constexpr (dimension_tag == _dimension_tag_ &&
                        MediumTag == _medium_tag_) {
            impl::compute_source_interaction<dimension_tag, wavefield, ngll,
                                             _medium_tag_, _property_tag_,
                                             _boundary_tag_>(assembly, istep);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
         BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                      COMPOSITE_STACEY_DIRICHLET)),
        {
          if constexpr (dimension_tag == _dimension_tag_ &&
                        MediumTag == _medium_tag_) {
            elements_updated += impl::compute_stiffness_interaction<
                dimension_tag, wavefield, ngll, _medium_tag_, _property_tag_,
                _boundary_tag_>(assembly, istep);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        {
          if constexpr (dimension_tag == _dimension_tag_ &&
                        MediumTag == _medium_tag_) {
            impl::divide_mass_matrix<dimension_tag, wavefield, _medium_tag_>(
                assembly);
          }
        })

    // FOR_EACH_IN_PRODUCT(
    //     (DIMENSION_TAG(DIM3), CONNECTION_TAG(WEAKLY_CONFORMING),
    //      INTERFACE_TAG(ELASTIC), BOUNDARY_TAG(NONE)),
    //     {
    //       constexpr auto self_medium =
    //           specfem::interface::attributes<_dimension_tag_,
    //                                          _interface_tag_>::self_medium();
    //       if constexpr (dimension_tag == _dimension_tag_ &&
    //                     self_medium == MediumTag) {
    //         impl::compute_coupling<_dimension_tag_, _connection_tag_,
    //         wavefield,
    //                                _interface_tag_,
    //                                _boundary_tag_>(assembly);
    //       }
    //     })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC),
         BOUNDARY_TAG(NONE)),
        {
          if constexpr (dimension_tag == _dimension_tag_ &&
                        MediumTag == _medium_tag_) {
            impl::compute_source_interaction<dimension_tag, wavefield, ngll,
                                             _medium_tag_, _property_tag_,
                                             _boundary_tag_>(assembly, istep);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC),
         BOUNDARY_TAG(NONE)),
        {
          if constexpr (dimension_tag == _dimension_tag_ &&
                        MediumTag == _medium_tag_) {
            elements_updated += impl::compute_stiffness_interaction<
                dimension_tag, wavefield, ngll, _medium_tag_, _property_tag_,
                _boundary_tag_>(assembly, istep);
          }
        })

    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)), {
      if constexpr (dimension_tag == _dimension_tag_ &&
                    MediumTag == _medium_tag_) {
        impl::divide_mass_matrix<dimension_tag, wavefield, _medium_tag_>(
            assembly);
      }
    })

    return elements_updated;
  }

  /**
   * @brief Initializes the mass matrix for the simulation
   *
   * This function initializes the mass matrix for the simulation. It computes
   * the mass matrix and inverts it for different MediumTag types.
   *
   * @param dt Time step for the simulation
   */
  void initialize(const type_real &dt) {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
         BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                      COMPOSITE_STACEY_DIRICHLET)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            impl::compute_mass_matrix<dimension_tag, wavefield, ngll,
                                      _medium_tag_, _property_tag_,
                                      _boundary_tag_>(dt, assembly);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            impl::invert_mass_matrix<dimension_tag, wavefield, _medium_tag_>(
                assembly);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC),
         BOUNDARY_TAG(NONE)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            impl::compute_mass_matrix<dimension_tag, wavefield, ngll,
                                      _medium_tag_, _property_tag_,
                                      _boundary_tag_>(dt, assembly);
          }
        })

    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)), {
      if constexpr (dimension_tag == _dimension_tag_) {
        impl::invert_mass_matrix<dimension_tag, wavefield, _medium_tag_>(
            assembly);
      }
    })

    return;
  }

  /**
   * @brief Computes the seismograms for the simulation
   *
   * This function computes the seismograms for the simulation. It is
   * specialized for different MediumTag types and properties.
   *
   * @param isig_step Time step for which the seismograms are computed
   */
  inline void compute_seismograms(const int &isig_step) {

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            impl::compute_seismograms<dimension_tag, wavefield, ngll,
                                      _medium_tag_, _property_tag_>(assembly,
                                                                    isig_step);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC_PSV), PROPERTY_TAG(ISOTROPIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            impl::compute_seismograms<dimension_tag, wavefield, ngll,
                                      _medium_tag_, _property_tag_>(assembly,
                                                                    isig_step);
          }
        })
  }

private:
  specfem::assembly::assembly<dimension_tag> assembly;
};

} // namespace kokkos_kernels
} // namespace specfem
