#ifndef _COMPUTE_FIELDS_IMPL_FIELD_IMPL_TPP_
#define _COMPUTE_FIELDS_IMPL_FIELD_IMPL_TPP_

#include "compute/fields/impl/field_impl.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace {
int compute_nglob(const specfem::kokkos::HostView3d<int> index_mapping) {
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

template <typename medium>
specfem::compute::impl::field_impl<medium>::field_impl(const int nglob,
                                                       const int nspec,
                                                       const int ngllz,
                                                       const int ngllx)
    : nglob(nglob), nspec(nspec),
      index_mapping("specfem::compute::fields::index_mapping", nspec, ngllz,
                    ngllx),
      h_index_mapping(Kokkos::create_mirror_view(index_mapping)),
      field("specfem::compute::fields::field", nglob, medium::components),
      h_field(Kokkos::create_mirror_view(field)),
      field_dot("specfem::compute::fields::field_dot", nglob,
                medium::components),
      h_field_dot(Kokkos::create_mirror_view(field_dot)),
      field_dot_dot("specfem::compute::fields::field_dot_dot", nglob,
                    medium::components),
      h_field_dot_dot(Kokkos::create_mirror_view(field_dot_dot)),
      mass_inverse("specfem::compute::fields::mass_inverse", nglob, nglob),
      h_mass_inverse(Kokkos::create_mirror_view(mass_inverse)) {}

template <typename medium>
specfem::compute::impl::field_impl<medium>::field_impl(
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties,
    Kokkos::View<int *, Kokkos::LayoutLeft,
                 specfem::kokkos::HostMemSpace>
        assembly_index_mapping) {

  const auto index_mapping = mesh.points.index_mapping;
  const auto element_type = properties.h_element_types;
  const int nspec = mesh.points.nspec;
  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  // Count the total number of distinct global indices for the medium
  int count = 0;

  for (int ispec = 0; ispec < nspec; ++ispec) {
    // increase the count only if current element is of the medium type
    if (element_type(ispec) == medium::value) {
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          const int index = index_mapping(ispec, iz, ix);
          // increase the count only if the global index is not already counted
          /// static_cast<int>(medium::value) is the index of the medium in the
          /// enum class
          if (assembly_index_mapping(index) ==
              -1) {
            assembly_index_mapping(index) =
                count;
            count++;
          }
        }
      }
    }
  }

  nglob = count;

  field = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field", nglob, medium::components);
  h_field = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(field));
  field_dot = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field_dot", nglob, medium::components);
  h_field_dot = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(field_dot));
  field_dot_dot = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field_dot_dot", nglob, medium::components);
  h_field_dot_dot =
      specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
          Kokkos::create_mirror_view(field_dot_dot));
  mass_inverse = specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::mass_inverse", nglob, medium::components);
  h_mass_inverse = specfem::kokkos::HostMirror2d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(mass_inverse));

  Kokkos::parallel_for(
      "specfem::compute::fields::field_impl::initialize_field",
      specfem::kokkos::HostRange(0, nglob), KOKKOS_LAMBDA(const int &iglob) {
        for (int icomp = 0; icomp < medium::components; ++icomp) {
          h_field(iglob, icomp) = 0.0;
          h_field_dot(iglob, icomp) = 0.0;
          h_field_dot_dot(iglob, icomp) = 0.0;
          h_mass_inverse(iglob, icomp) = 0.0;
        }
      });

  Kokkos::deep_copy(field, h_field);
  Kokkos::deep_copy(field_dot, h_field_dot);
  Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  Kokkos::deep_copy(mass_inverse, h_mass_inverse);

  return;
}

template <typename medium>
template <specfem::sync::kind sync>
void specfem::compute::impl::field_impl<medium>::sync_fields() const {
  if constexpr (sync == specfem::sync::kind::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
    Kokkos::deep_copy(h_field_dot, field_dot);
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
    Kokkos::deep_copy(h_mass_inverse, mass_inverse);
  } else if constexpr (sync == specfem::sync::kind::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
    Kokkos::deep_copy(field_dot, h_field_dot);
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
    Kokkos::deep_copy(mass_inverse, h_mass_inverse);
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
