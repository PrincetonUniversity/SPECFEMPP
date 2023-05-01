#include "kokkos_abstractions.h"
#include "source_time_function/interface.hpp"

KOKKOS_INLINE_FUNCTION
specfem::forcing_function::stf *assign_stf(std::string forcing_type,
                                           type_real f0, type_real tshift,
                                           type_real factor, type_real dt,
                                           bool use_trick_for_better_pressure) {

  specfem::forcing_function::stf *forcing_function;
  if (forcing_type == "Dirac") {
    forcing_function = (specfem::forcing_function::stf *)
        Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(
            sizeof(specfem::forcing_function::Dirac));

    f0 = 1.0 / (10.0 * dt);

    Kokkos::parallel_for(
        "specfem::sources::moment_tensor::moment_tensor::allocate_stf",
        specfem::kokkos::DeviceRange(0, 1), KOKKOS_LAMBDA(const int &) {
          new (forcing_function) specfem::forcing_function::Dirac(
              f0, tshift, factor, use_trick_for_better_pressure);
        });

    Kokkos::fence();
  }

  return forcing_function;
}

KOKKOS_INLINE_FUNCTION
specfem::forcing_function::stf *
assign_dirac(YAML::Node &Dirac, type_real dt,
             bool use_trick_for_better_pressure) {

  specfem::forcing_function::stf *forcing_function;
  forcing_function = (specfem::forcing_function::stf *)
      Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(
          sizeof(specfem::forcing_function::Dirac));

  type_real f0 = 1.0 / (10.0 * dt);
  type_real tshift = Dirac["tshift"].as<type_real>();
  type_real factor = Dirac["factor"].as<type_real>();

  Kokkos::parallel_for(
      "specfem::sources::moment_tensor::moment_tensor::allocate_stf",
      specfem::kokkos::DeviceRange(0, 1), KOKKOS_LAMBDA(const int &) {
        new (forcing_function) specfem::forcing_function::Dirac(
            f0, tshift, factor, use_trick_for_better_pressure);
      });

  Kokkos::fence();

  return forcing_function;
}
