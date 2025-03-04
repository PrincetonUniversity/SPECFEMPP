#ifndef _COMPUTE_FIELDS_IMPL_FIELD_IMPL_TPP_
#define _COMPUTE_FIELDS_IMPL_FIELD_IMPL_TPP_

#include "compute/element_types/element_types.hpp"
#include "compute/fields/impl/field_impl.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
specfem::compute::impl::field_impl<DimensionType, MediumTag>::field_impl(
    const int nglob)
    : nglob(nglob), field("specfem::compute::fields::field", nglob, components),
      h_field(Kokkos::create_mirror_view(field)),
      field_dot("specfem::compute::fields::field_dot", nglob, components),
      h_field_dot(Kokkos::create_mirror_view(field_dot)),
      field_dot_dot("specfem::compute::fields::field_dot_dot", nglob,
                    components),
      h_field_dot_dot(Kokkos::create_mirror_view(field_dot_dot)),
      mass_inverse("specfem::compute::fields::mass_inverse", nglob, components),
      h_mass_inverse(Kokkos::create_mirror_view(mass_inverse)) {}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
specfem::compute::impl::field_impl<DimensionType, MediumTag>::field_impl(
    const specfem::compute::mesh &mesh,
    const specfem::compute::element_types &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
        assembly_index_mapping) {

  const auto index_mapping = mesh.points.h_index_mapping;
  const int nspec = mesh.points.nspec;
  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  // Count the total number of distinct global indices for the medium
  int count = 0;
  using simd = specfem::datatype::simd<type_real, true>;

  constexpr int chunk_size = specfem::parallel_config::storage_chunk_size;
  int nchunks = nspec / chunk_size;
  int iloc = 0;
  for (int ichunk = 0; ichunk < nchunks; ichunk++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk * chunk_size + ielement;
          const auto medium = element_types.get_medium_tag(ispec);
          if (medium == MediumTag) {
            const int index = index_mapping(ispec, iz, ix); // get global index
            // increase the count only if the global index is not already
            // counted
            /// static_cast<int>(medium::value) is the index of the medium in
            /// the enum class
            if (assembly_index_mapping(index) == -1) {
              assembly_index_mapping(index) = count;
              count++;
            }
          }
        }
      }
    }
  }

  nglob = count;

  field = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field", nglob, components);
  h_field = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(field));
  field_dot = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field_dot", nglob, components);
  h_field_dot = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(field_dot));
  field_dot_dot = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field_dot_dot", nglob, components);
  h_field_dot_dot =
      specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
          Kokkos::create_mirror_view(field_dot_dot));
  mass_inverse = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::mass_inverse", nglob, components);
  h_mass_inverse = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(mass_inverse));

  Kokkos::parallel_for("specfem::compute::fields::field_impl::initialize_field",
                       specfem::kokkos::HostRange(0, nglob),
                       [=](const int &iglob) {
                         for (int icomp = 0; icomp < components; ++icomp) {
                           h_field(iglob, icomp) = 0.0;
                           h_field_dot(iglob, icomp) = 0.0;
                           h_field_dot_dot(iglob, icomp) = 0.0;
                           h_mass_inverse(iglob, icomp) = 0.0;
                         }
                       });

  Kokkos::fence();

  Kokkos::deep_copy(field, h_field);
  Kokkos::deep_copy(field_dot, h_field_dot);
  Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  Kokkos::deep_copy(mass_inverse, h_mass_inverse);

  return;
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
template <specfem::sync::kind sync>
void specfem::compute::impl::field_impl<DimensionType, MediumTag>::sync_fields()
    const {
  if constexpr (sync == specfem::sync::kind::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
    Kokkos::deep_copy(h_field_dot, field_dot);
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
  } else if constexpr (sync == specfem::sync::kind::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
    Kokkos::deep_copy(field_dot, h_field_dot);
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  }
}

#endif /* _COMPUTE_FIELDS_IMPL_FIELD_IMPL_TPP_ */

// template <typename medium>
//   KOKKOS_INLINE_FUNCTION type_real &specfem::compute::(const int &iglob,
//   const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.field(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.field(index, icomp);
//     }
//   }

//   template <typename medium>
//   inline type_real &h_field(const int &iglob, const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.h_field(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.h_field(index, icomp);
//     }
//   }

//   template <typename medium>
//   KOKKOS_INLINE_FUNCTION type_real &field_dot(const int &iglob,
//                                               const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.field_dot(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.field_dot(index, icomp);
//     }
//   }

//   template <typename medium>
//   inline type_real &h_field_dot(const int &iglob, const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.h_field_dot(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.h_field_dot(index, icomp);
//     }
//   }

//   template <typename medium>
//   KOKKOS_INLINE_FUNCTION type_real &field_dot_dot(const int &iglob,
//                                                   const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.field_dot_dot(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.field_dot_dot(index, icomp);
//     }
//   }

//   template <typename medium>
//   inline type_real &h_field_dot_dot(const int &iglob, const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.h_field_dot_dot(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.h_field_dot_dot(index, icomp);
//     }
//   }

//   template <typename medium>
//   KOKKOS_INLINE_FUNCTION type_real &mass_inverse(const int &iglob,
//                                                  const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.mass_inverse(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.mass_inverse(index, icomp);
//     }
//   }

//   template <typename medium>
//   inline type_real &h_mass_inverse(const int &iglob, const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.h_mass_inverse(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.h_mass_inverse(index, icomp);
//     }
//   }
