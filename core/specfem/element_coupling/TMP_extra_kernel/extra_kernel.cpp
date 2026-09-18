#pragma once

#include "specfem/element_coupling/TMP_extra_kernel/extra_kernel.hpp"
#include "specfem/element.hpp"
#include "specfem/macros/compute_instantiation_macros.hpp"
#include "specfem/simulation.hpp"
#include "specfem/tags.hpp"

#include "acoustic_elastic.tpp"

using specfem::element::dimension_tag;
using specfem::element::medium_tag;
using specfem::simulation::field_type;

template <int NGLL, typename Tags>
void specfem::element_coupling::TMP_extra_kernel::execute(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {
  compute_coupling_extra_kernel<NGLL, Tags>().execute(assembly);
}

template <int NGLL, typename Tags>
void _impl_expand_like_compute_coupling(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {

  constexpr auto WavefieldType = Tags::wavefield_tag;

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          CONNECTION_SET(weakly_conforming, nonconforming) *
          INTERFACE_SET(elastic_acoustic, acoustic_elastic) *
          BOUNDARY_SET(none, acoustic_free_surface, stacey,
                       composite_stacey_dirichlet) *
          FLUX_SCHEME_SET(natural),
      [&]<typename ElementTags>() {
        constexpr auto self_medium = specfem::element_coupling::attributes<
            ElementTags::dimension_tag,
            ElementTags::interface_tag>::self_medium();
        if constexpr (self_medium == Tags::medium_tag) {
          specfem::element_coupling::TMP_extra_kernel::execute<
              NGLL,
              specfem::tags::Tags<
                  ElementTags::dimension_tag, ElementTags::connection_tag,
                  WavefieldType, ElementTags::interface_tag,
                  ElementTags::boundary_tag, ElementTags::flux_scheme_tag>>(
              assembly);
        }
      });
}

#define SIGNATURE(NGLL, DIM, WF, MED)                                          \
  template void                                                                \
  _impl_expand_like_compute_coupling<NGLL, specfem::tags::Tags<DIM, WF, MED>>( \
      const specfem::assembly::assembly<DIM> &);
SPECFEM_COMPUTE_COMBINATIONS(SIGNATURE)
#undef SIGNATURE
