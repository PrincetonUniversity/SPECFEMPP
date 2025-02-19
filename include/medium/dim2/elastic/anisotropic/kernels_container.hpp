#pragma once
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/coordinates.hpp"
#include "point/kernels.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <>
class kernels_container<specfem::element::medium_tag::elastic_sv,
                        specfem::element::property_tag::anisotropic> {
public:
  constexpr static auto value_type = specfem::element::medium_tag::elastic_sv;
  constexpr static auto property_type =
      specfem::element::property_tag::anisotropic;
  int nspec;
  int ngllz;
  int ngllx;

  using ViewType = Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace>;

  ViewType rho;
  ViewType::HostMirror h_rho;
  ViewType c11;
  ViewType::HostMirror h_c11;
  ViewType c13;
  ViewType::HostMirror h_c13;
  ViewType c15;
  ViewType::HostMirror h_c15;
  ViewType c33;
  ViewType::HostMirror h_c33;
  ViewType c35;
  ViewType::HostMirror h_c35;
  ViewType c55;
  ViewType::HostMirror h_c55;

  kernels_container() = default;

  kernels_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        rho("specfem::compute::properties::rho", nspec, ngllz, ngllx),
        h_rho(Kokkos::create_mirror_view(rho)),
        c11("specfem::compute::properties::c11", nspec, ngllz, ngllx),
        h_c11(Kokkos::create_mirror_view(c11)),
        c13("specfem::compute::properties::c13", nspec, ngllz, ngllx),
        h_c13(Kokkos::create_mirror_view(c13)),
        c15("specfem::compute::properties::c15", nspec, ngllz, ngllx),
        h_c15(Kokkos::create_mirror_view(c15)),
        c33("specfem::compute::properties::c33", nspec, ngllz, ngllx),
        h_c33(Kokkos::create_mirror_view(c33)),
        c35("specfem::compute::properties::c35", nspec, ngllz, ngllx),
        h_c35(Kokkos::create_mirror_view(c35)),
        c55("specfem::compute::properties::c55", nspec, ngllz, ngllx),
        h_c55(Kokkos::create_mirror_view(c55)) {

    initialize();
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void load_device_kernels(
      const specfem::point::index<PointKernelType::dimension> &index,
      PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    kernels.rho = rho(ispec, iz, ix);
    kernels.c11 = c11(ispec, iz, ix);
    kernels.c13 = c13(ispec, iz, ix);
    kernels.c15 = c15(ispec, iz, ix);
    kernels.c33 = c33(ispec, iz, ix);
    kernels.c35 = c35(ispec, iz, ix);
    kernels.c55 = c55(ispec, iz, ix);
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void load_device_kernels(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    Kokkos::Experimental::where(mask, kernels.rho)
        .copy_from(&rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c11)
        .copy_from(&c11(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c13)
        .copy_from(&c13(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c15)
        .copy_from(&c15(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c33)
        .copy_from(&c33(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c35)
        .copy_from(&c35(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c55)
        .copy_from(&c55(ispec, iz, ix), tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  void
  load_host_kernels(specfem::point::index<PointKernelType::dimension> &index,
                    PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    kernels.rho = h_rho(ispec, iz, ix);
    kernels.c11 = h_c11(ispec, iz, ix);
    kernels.c13 = h_c13(ispec, iz, ix);
    kernels.c15 = h_c15(ispec, iz, ix);
    kernels.c33 = h_c33(ispec, iz, ix);
    kernels.c35 = h_c35(ispec, iz, ix);
    kernels.c55 = h_c55(ispec, iz, ix);
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  void load_host_kernels(
      specfem::point::simd_index<PointKernelType::dimension> &index,
      PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    Kokkos::Experimental::where(mask, kernels.rho)
        .copy_from(&h_rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c11)
        .copy_from(&h_c11(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c13)
        .copy_from(&h_c13(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c15)
        .copy_from(&h_c15(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c33)
        .copy_from(&h_c33(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c35)
        .copy_from(&h_c35(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c55)
        .copy_from(&h_c55(ispec, iz, ix), tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void update_kernels_on_device(
      const specfem::point::index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    rho(ispec, iz, ix) = kernels.rho;
    c11(ispec, iz, ix) = kernels.c11;
    c13(ispec, iz, ix) = kernels.c13;
    c15(ispec, iz, ix) = kernels.c15;
    c33(ispec, iz, ix) = kernels.c33;
    c35(ispec, iz, ix) = kernels.c35;
    c55(ispec, iz, ix) = kernels.c55;
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void update_kernels_on_device(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    Kokkos::Experimental::where(mask, kernels.rho)
        .copy_to(&rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c11)
        .copy_to(&c11(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c13)
        .copy_to(&c13(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c15)
        .copy_to(&c15(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c33)
        .copy_to(&c33(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c35)
        .copy_to(&c35(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c55)
        .copy_to(&c55(ispec, iz, ix), tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  void update_kernels_on_host(
      const specfem::point::index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    h_rho(ispec, iz, ix) = kernels.rho;
    h_c11(ispec, iz, ix) = kernels.c11;
    h_c13(ispec, iz, ix) = kernels.c13;
    h_c15(ispec, iz, ix) = kernels.c15;
    h_c33(ispec, iz, ix) = kernels.c33;
    h_c35(ispec, iz, ix) = kernels.c35;
    h_c55(ispec, iz, ix) = kernels.c55;
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  void update_kernels_on_host(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;
    Kokkos::Experimental::where(mask, kernels.rho)
        .copy_to(&h_rho(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c11)
        .copy_to(&h_c11(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c13)
        .copy_to(&h_c13(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c15)
        .copy_to(&h_c15(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c33)
        .copy_to(&h_c33(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c35)
        .copy_to(&h_c35(ispec, iz, ix), tag_type());
    Kokkos::Experimental::where(mask, kernels.c55)
        .copy_to(&h_c55(ispec, iz, ix), tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void add_kernels_on_device(
      const specfem::point::index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    rho(ispec, iz, ix) += kernels.rho;
    c11(ispec, iz, ix) += kernels.c11;
    c13(ispec, iz, ix) += kernels.c13;
    c15(ispec, iz, ix) += kernels.c15;
    c33(ispec, iz, ix) += kernels.c33;
    c35(ispec, iz, ix) += kernels.c35;
    c55(ispec, iz, ix) += kernels.c55;
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  KOKKOS_INLINE_FUNCTION void add_kernels_on_device(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    simd_type lhs;

    Kokkos::Experimental::where(mask, lhs).copy_from(&rho(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.rho;
    Kokkos::Experimental::where(mask, lhs).copy_to(&rho(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&c11(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c11;
    Kokkos::Experimental::where(mask, lhs).copy_to(&c11(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&c13(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c13;
    Kokkos::Experimental::where(mask, lhs).copy_to(&c13(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&c15(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c15;
    Kokkos::Experimental::where(mask, lhs).copy_to(&c15(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&c33(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c33;
    Kokkos::Experimental::where(mask, lhs).copy_to(&c33(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&c35(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c35;
    Kokkos::Experimental::where(mask, lhs).copy_to(&c35(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&c55(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c55;
    Kokkos::Experimental::where(mask, lhs).copy_to(&c55(ispec, iz, ix),
                                                   tag_type());
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<!PointKernelType::simd::using_simd, int> = 0>
  void add_kernels_on_host(
      const specfem::point::index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    h_rho(ispec, iz, ix) += kernels.rho;
    h_c11(ispec, iz, ix) += kernels.c11;
    h_c13(ispec, iz, ix) += kernels.c13;
    h_c15(ispec, iz, ix) += kernels.c15;
    h_c33(ispec, iz, ix) += kernels.c33;
    h_c35(ispec, iz, ix) += kernels.c35;
    h_c55(ispec, iz, ix) += kernels.c55;
  }

  template <
      typename PointKernelType,
      typename std::enable_if_t<PointKernelType::simd::using_simd, int> = 0>
  void add_kernels_on_host(
      const specfem::point::simd_index<PointKernelType::dimension> &index,
      const PointKernelType &kernels) const {

    static_assert(PointKernelType::medium_tag == value_type);
    static_assert(PointKernelType::property_tag == property_type);

    using simd_type = typename PointKernelType::simd::datatype;
    using mask_type = typename PointKernelType::simd::mask_type;
    using tag_type = typename PointKernelType::simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    simd_type lhs;

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_rho(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.rho;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_rho(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_c11(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c11;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_c11(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_c13(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c13;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_c13(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_c15(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c15;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_c15(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_c33(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c33;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_c33(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_c35(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c35;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_c35(ispec, iz, ix),
                                                   tag_type());

    Kokkos::Experimental::where(mask, lhs).copy_from(&h_c55(ispec, iz, ix),
                                                     tag_type());
    lhs += kernels.c55;
    Kokkos::Experimental::where(mask, lhs).copy_to(&h_c55(ispec, iz, ix),
                                                   tag_type());
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_c11, c11);
    Kokkos::deep_copy(h_c13, c13);
    Kokkos::deep_copy(h_c15, c15);
    Kokkos::deep_copy(h_c33, c33);
    Kokkos::deep_copy(h_c35, c35);
    Kokkos::deep_copy(h_c55, c55);
  }

  void copy_to_device() {
    Kokkos::deep_copy(rho, h_rho);
    Kokkos::deep_copy(c11, h_c11);
    Kokkos::deep_copy(c13, h_c13);
    Kokkos::deep_copy(c15, h_c15);
    Kokkos::deep_copy(c33, h_c33);
    Kokkos::deep_copy(c35, h_c35);
    Kokkos::deep_copy(c55, h_c55);
  }

  void initialize() {
    Kokkos::parallel_for(
        "specfem::medium::elastic::anisotropic::initialize",
        Kokkos::MDRangePolicy<Kokkos::Rank<3> >({ 0, 0, 0 },
                                                { nspec, ngllz, ngllx }),
        KOKKOS_CLASS_LAMBDA(const int ispec, const int iz, const int ix) {
          this->rho(ispec, iz, ix) = 0.0;
          this->c11(ispec, iz, ix) = 0.0;
          this->c13(ispec, iz, ix) = 0.0;
          this->c15(ispec, iz, ix) = 0.0;
          this->c33(ispec, iz, ix) = 0.0;
          this->c35(ispec, iz, ix) = 0.0;
          this->c55(ispec, iz, ix) = 0.0;
        });
  }
};

} // namespace medium
} // namespace specfem
