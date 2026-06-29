#include "impl/compute_coupling.hpp"
#include "impl/compute_source_interaction.hpp"
#include "impl/compute_stiffness_interaction.hpp"
#include "impl/divide_mass_matrix.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/mpi/dim3/mpi.hpp"
#include "specfem/data_access/data_class.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/tags.hpp"
#include "update_wavefields.hpp"

namespace specfem::compute {

/**
 * @brief Updates the wavefield for a given medium
 *
 * This function updates the wavefield for a given medium type. It computes
 * the coupling, source interaction, stiffness interaction, and divides the
 * mass matrix. The function is specialized for different medium types and
 * properties.
 *
 * @tparam WavefieldType Type of the wavefield
 * @tparam DimensionTag Dimension tag
 * @tparam NGLL Number of GLL points
 * @tparam MediumTag Medium for which the wacefield is updated
 * @param assembly The assembly object containing the mesh
 * @param istep Time step for which the wavefield is updated
 * @return int Number of elements updated
 */
template <int NGLL, typename Tags>
int update_wavefields(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const int istep) {
  constexpr auto backward = specfem::simulation::field_type::backward;
  constexpr auto wavefield = Tags::wavefield_tag;

  impl::compute_coupling<NGLL, Tags>(assembly);

  // Fortran backward source replay uses NSTEP - it + 1. Because C++ iterates
  // backward with zero-based istep values, this is the previous source sample
  // relative to the current reconstructed-field stiffness/Stacey update.
  const int source_istep = (wavefield == backward && istep > 0) ? istep - 1
                                                               : istep;
  impl::compute_source_interaction<NGLL, Tags>(assembly, source_istep);

  int elements_updated = 0;

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
      specfem::tag_dispatch::medium_set<Tags::medium_tag>{} *
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
      ATTENUATION_SET(none, constant_isotropic) *
      BOUNDARY_SET(none, stacey, acoustic_free_surface,
                   composite_stacey_dirichlet), [&]<typename ElementTags>() {
    elements_updated +=
        impl::compute_stiffness_interaction<NGLL, specfem::tags::expand<ElementTags, Tags::wavefield_tag>>(
            assembly, istep);
  });

  if constexpr (Tags::dimension_tag ==
                specfem::element::dimension_tag::dim3) {
    constexpr auto medium = Tags::medium_tag;
    constexpr auto acceleration =
        specfem::data_access::DataClassType::acceleration;

    if constexpr (medium == specfem::element::medium_tag::acoustic ||
                  medium == specfem::element::medium_tag::elastic) {
      auto mpi_buf = assembly.mpi_interfaces.template create_mpi_buffer<
          wavefield, medium, acceleration>();
      auto &field = [&]() -> auto & {
        if constexpr (wavefield == specfem::simulation::field_type::forward)
          return assembly.fields.forward;
        else if constexpr (wavefield ==
                           specfem::simulation::field_type::adjoint)
          return assembly.fields.adjoint;
        else
          return assembly.fields.backward;
      }();

      mpi_buf.pack(field);
      mpi_buf.receive();
      mpi_buf.send();
      mpi_buf.wait();
      mpi_buf.unpack(field);
    }
  }

  impl::divide_mass_matrix<NGLL, Tags>(assembly);
  return elements_updated;
}

} // namespace specfem::compute
