#ifndef _COMPUTE_FIELDS_SIMULATION_FIELD_TPP_
#define _COMPUTE_FIELDS_SIMULATION_FIELD_TPP_

#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/impl/field_impl.tpp"
#include "compute/fields/simulation_field.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <specfem::simulation::type simulation>
specfem::compute::simulation_field<simulation>::simulation_field(
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties) {

  nglob = compute_nglob(mesh.points.index_mapping);

  assembly_index_mapping =
      Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
                   specfem::kokkos::DevMemSpace>(
          "specfem::compute::simulation_field::index_mapping", nglob);

  h_assembly_index_mapping =
      Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
                   specfem::kokkos::HostMemSpace>(
          Kokkos::create_mirror_view(assembly_index_mapping));

  for (int iglob = 0; iglob < nglob; iglob++) {
    for (int itype = 0; itype < specfem::element::ntypes; itype++) {
      h_assembly_index_mapping(iglob, itype) = -1;
    }
  }

  auto acoustic_index = Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
                                        static_cast<int>(acoustic_type::medium_tag));

  auto elastic_index = Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
                                       static_cast<int>(elastic_type::medium_tag));

  elastic = specfem::compute::impl::field_impl<elastic_type>(mesh, properties,
                                                             elastic_index);

  acoustic = specfem::compute::impl::field_impl<acoustic_type>(mesh, properties,
                                                               acoustic_index);

  Kokkos::deep_copy(assembly_index_mapping, h_assembly_index_mapping);

  return;
}

// template <typename simulation>
// template <typename medium>
// KOKKOS_INLINE_FUNCTION type_real &
// specfem::compute::simulation_field<simulation>::field(const int &iglob,
//                                                       const int &icomp) {
//   if constexpr (std::is_same_v<medium, elastic_type>) {
//     int index = assembly_index_mapping(iglob,
//     static_cast<int>(medium::value)); return elastic.field(index, icomp);
//   } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//     int index = assembly_index_mapping(iglob,
//     static_cast<int>(medium::value)); return acoustic.field(index, icomp);
//   }
// }

// template <typename simulation>
// template <typename medium>
// inline type_real &
// specfem::compute::simulation_field<simulation>::h_field(const int &iglob,
//                                                         const int &icomp) {
//   if constexpr (std::is_same_v<medium, elastic_type>) {
//     int index =
//         h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//     return elastic.h_field(index, icomp);
//   } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//     int index =
//         h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//     return acoustic.h_field(index, icomp);
//   }
// }

// template <typename simulation>
// template <typename medium>
// KOKKOS_INLINE_FUNCTION type_real &
// specfem::compute::simulation_field<simulation>::field_dot(const int &iglob,
//                                                           const int &icomp) {
//   if constexpr (std::is_same_v<medium, elastic_type>) {
//     int index = assembly_index_mapping(iglob,
//     static_cast<int>(medium::value)); return elastic.field_dot(index, icomp);
//   } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//     int index = assembly_index_mapping(iglob,
//     static_cast<int>(medium::value)); return acoustic.field_dot(index,
//     icomp);
//   }
// }

// template <typename simulation>
// template <typename medium>
// inline type_real &
// specfem::compute::simulation_field<simulation>::h_field_dot(const int &iglob,
//                                                             const int &icomp)
//                                                             {
//   if constexpr (std::is_same_v<medium, elastic_type>) {
//     int index =
//         h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//     return elastic.h_field_dot(index, icomp);
//   } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//     int index =
//         h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//     return acoustic.h_field_dot(index, icomp);
//   }
// }

// template <typename simulation>
// template <typename medium>
// KOKKOS_INLINE_FUNCTION type_real &
// specfem::compute::simulation_field<simulation>::field_dot_dot(
//     const int &iglob, const int &icomp) {
//   if constexpr (std::is_same_v<medium, elastic_type>) {
//     int index = assembly_index_mapping(iglob,
//     static_cast<int>(medium::value)); return elastic.field_dot_dot(index,
//     icomp);
//   } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//     int index = assembly_index_mapping(iglob,
//     static_cast<int>(medium::value)); return acoustic.field_dot_dot(index,
//     icomp);
//   }
// }

// template <typename simulation>
// template <typename medium>
// inline type_real &
// specfem::compute::simulation_field<simulation>::h_field_dot_dot(
//     const int &iglob, const int &icomp) {
//   if constexpr (std::is_same_v<medium, elastic_type>) {
//     int index =
//         h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//     return elastic.h_field_dot_dot(index, icomp);
//   } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//     int index =
//         h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//     return acoustic.h_field_dot_dot(index, icomp);
//   }
// }

// template <typename simulation>
// template <typename medium>
// KOKKOS_INLINE_FUNCTION type_real &
// specfem::compute::simulation_field<simulation>::mass_inverse(const int
// &iglob,
//                                                              const int
//                                                              &icomp) {
//   if constexpr (std::is_same_v<medium, elastic_type>) {
//     int index = assembly_index_mapping(iglob,
//     static_cast<int>(medium::value)); return elastic.mass_inverse(index,
//     icomp);
//   } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//     int index = assembly_index_mapping(iglob,
//     static_cast<int>(medium::value)); return acoustic.mass_inverse(index,
//     icomp);
//   }
// }

#endif /* _COMPUTE_FIELDS_SIMULATION_FIELD_TPP_ */
