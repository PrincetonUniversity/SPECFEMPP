#ifndef _COMPUTE_FIELDS_SIMULATION_FIELD_TPP_
#define _COMPUTE_FIELDS_SIMULATION_FIELD_TPP_

#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/impl/field_impl.tpp"
#include "compute/fields/simulation_field.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace {
template <typename ViewType>
int compute_nglob(const ViewType index_mapping) {
  const int nspec = index_mapping.extent(0);
  const int ngllz = index_mapping.extent(1);
  const int ngllx = index_mapping.extent(2);

  int nglob;
  Kokkos::parallel_reduce(
      "specfem::utils::compute_nglob",
      specfem::kokkos::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int ispec, const int iz, const int ix, int &l_nglob) {
        l_nglob = l_nglob > index_mapping(ispec, iz, ix)
                      ? l_nglob
                      : index_mapping(ispec, iz, ix);
      },
      Kokkos::Max<int>(nglob));

  return nglob + 1;
}
} // namespace

template <specfem::wavefield::type WavefieldType>
specfem::compute::simulation_field<WavefieldType>::simulation_field(
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties) {

  nglob = compute_nglob(mesh.points.h_index_mapping);

  this->nspec = mesh.points.nspec;
  this->ngllz = mesh.points.ngllz;
  this->ngllx = mesh.points.ngllx;
  this->index_mapping = mesh.points.index_mapping;
  this->h_index_mapping = mesh.points.h_index_mapping;

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

  auto acoustic_index =
      Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
                      static_cast<int>(acoustic_type::medium_tag));

  auto elastic_index =
      Kokkos::subview(h_assembly_index_mapping, Kokkos::ALL,
                      static_cast<int>(elastic_type::medium_tag));

  elastic =
      specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
                                         specfem::element::medium_tag::elastic>(
          mesh, properties, elastic_index);

  acoustic = specfem::compute::impl::field_impl<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>(
      mesh, properties, acoustic_index);

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
