#include "periodic_tasks/compute_energy.hpp"
#include "kokkos_kernels/impl/compute_energy.hpp"

template <int NGLL>
type_real specfem::periodic_tasks::compute_energy<NGLL>::compute() const {

  type_real energy = 0.0;
  constexpr auto dimension = specfem::dimension::type::dim2;

#define CALL_COMPUTE_ENERGY_FUNCTION(DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)  \
  energy += specfem::kokkos_kernels::impl::compute_energy<                     \
      GET_TAG(DIMENSION_TAG), specfem::wavefield::simulation_field::forward,   \
      ngll, GET_TAG(MEDIUM_TAG), GET_TAG(PROPERTY_TAG)>(assembly);

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      CALL_COMPUTE_ENERGY_FUNCTION,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH,
                MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef CALL_COMPUTE_ENERGY_FUNCTION

  return energy;
}

template <int NGLL> void specfem::periodic_tasks::compute_energy<NGLL>::run() {
  type_real energy = this->compute();
  std::cout << "Energy at timestep " << this->m_istep << ": " << energy
            << std::endl;
}

template class specfem::periodic_tasks::compute_energy<5>;
template class specfem::periodic_tasks::compute_energy<8>;
