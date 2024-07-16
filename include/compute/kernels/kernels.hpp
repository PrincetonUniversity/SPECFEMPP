#ifndef _SPECFEM_COMPUTE_KERNELS_HPP_
#define _SPECFEM_COMPUTE_KERNELS_HPP_

#include "enumerations/medium.hpp"
#include "impl/material_kernels.hpp"
#include "mesh/materials/interface.hpp"
#include "point/coordinates.hpp"
#include "point/kernels.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
struct kernels {
  int nspec;
  int ngllz;
  int ngllx;
  specfem::kokkos::DeviceView1d<specfem::element::medium_tag> element_types;
  specfem::kokkos::DeviceView1d<specfem::element::property_tag>
      element_property; ///< Element properties
  specfem::kokkos::HostMirror1d<specfem::element::medium_tag> h_element_types;
  specfem::kokkos::HostMirror1d<specfem::element::property_tag>
      h_element_property; ///< Element properties
  specfem::kokkos::DeviceView1d<int> property_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_property_index_mapping;

  specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>
      elastic_isotropic;

  specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>
      acoustic_isotropic;

  kernels() = default;

  kernels(const int nspec, const int ngllz, const int ngllx,
          const specfem::mesh::materials &materials);

  void copy_to_host() {
    Kokkos::deep_copy(h_element_types, element_types);
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    elastic_isotropic.copy_to_host();
    acoustic_isotropic.copy_to_host();
  }
};

template <typename PointKernelType, typename IndexType>
KOKKOS_FUNCTION void load_on_device(const IndexType &index,
                                    const kernels &kernels,
                                    PointKernelType &point_kernels) {
  const int ispec = kernels.property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.load_device_kernels(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.load_device_kernels(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

  return;
}

template <typename PointKernelType, typename IndexType>
void load_on_host(const IndexType &index, const kernels &kernels,
                  PointKernelType &point_kernels) {
  const int ispec = kernels.h_property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.load_host_kernels(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.load_host_kernels(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

  return;
}

template <typename PointKernelType, typename IndexType>
void store_on_host(const IndexType &index, const PointKernelType &point_kernels,
                   const kernels &kernels) {
  const int ispec = kernels.h_property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.update_kernels_on_host(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.update_kernels_on_host(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

  return;
}

template <typename PointKernelType, typename IndexType>
KOKKOS_FUNCTION void store_on_device(const IndexType &index,
                                     const PointKernelType &point_kernels,
                                     const kernels &kernels) {
  const int ispec = kernels.property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.update_kernels_on_device(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.update_kernels_on_device(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

  return;
}

template <typename IndexType, typename PointKernelType>
KOKKOS_FUNCTION void add_on_device(const IndexType &index,
                                   const PointKernelType &point_kernels,
                                   const kernels &kernels) {

  const int ispec = kernels.property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.add_kernels_on_device(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.add_kernels_on_device(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

  return;
}

template <typename IndexType, typename PointKernelType>
void add_on_host(const IndexType &index, const PointKernelType &point_kernels,
                 const kernels &kernels) {
  const int ispec = kernels.h_property_index_mapping(index.ispec);

  constexpr auto MediumTag = PointKernelType::medium_tag;
  constexpr auto PropertyTag = PointKernelType::property_tag;

  IndexType l_index = index;
  l_index.ispec = ispec;

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    kernels.elastic_isotropic.add_kernels_on_host(l_index, point_kernels);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    kernels.acoustic_isotropic.add_kernels_on_host(l_index, point_kernels);
  } else {
    static_assert("Material type not implemented");
  }

  return;
}

} // namespace compute
} // namespace specfem

#endif /* _SPECFEM_COMPUTE_KERNELS_HPP_ */
